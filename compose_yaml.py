from pathlib import Path

import yaml

BASE_CONFIG = Path("config/dromedario_sft.yaml")
MODEL_PATH = Path("config/models")
DATA_PATH = Path("config/data")
OUTPUT_PATH = Path("_configs")


def merge_yamls(*paths: Path) -> dict:
    merged = {}
    for p in paths:
        with open(p) as f:
            merged.update(yaml.safe_load(f))
    merged["run_name"] = f"{merged['codename']}-{merged['dataset']}"
    merged["output_dir"] = merged["output_dir"].rstrip("/") + f"/{merged['run_name']}"
    del merged["codename"]
    return merged


def main():
    OUTPUT_PATH.mkdir(exist_ok=True)

    model_configs = sorted(MODEL_PATH.glob("*.yaml"))
    data_configs = sorted(DATA_PATH.glob("*.yaml"))

    dataset_dir = set()
    for model_conf in model_configs:
        for data_conf in data_configs:
            job_name = f"{model_conf.stem}___{data_conf.stem}"

            config = merge_yamls(model_conf, data_conf, BASE_CONFIG)
            config["resume_from_checkpoint"] = None
            dataset_dir.add(config["dataset_dir"])
            out_path = OUTPUT_PATH / f"{job_name}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(config, f, sort_keys=False)

            print(f"llamafactory-cli train {out_path}")
            print(
                f'Per GPU batch size: {config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]}'
            )

    print("Remember to copy config/dataset_info.json to the dataset directories:")
    for dir in dataset_dir:
        print(f"  - {dir}")


if __name__ == "__main__":
    main()
