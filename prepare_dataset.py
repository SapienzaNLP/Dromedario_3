from pathlib import Path

import numpy as np
import ujson
from datasets import load_dataset
from tap import Tap
from tqdm import tqdm

TULU3_IT_NAME = "sapienzanlp/Dromedario_3"

EXCLUDED_SOURCES = {
    "ai2-adapt-dev/tulu_hard_coded_repeated_10",
}

ROLE_MAP = {
    "user": "human",
    "assistant": "gpt",
    "system": "system",
}

TASKS_TO_KEEP = [
    "mathematics",
    "text to code",
    "code to text",
    "program execution",
    "question answering",
    "explanation",
    "text completion",
    "story composition",
    "textual entailment",
    "misc.",
    "sentence composition",
    "sentiment analysis",
    "summarization",
    "question generation",
    "dialogue generation",
    "fact verification",
    "commonsense classification",
    "information extraction",
    "text categorization",
    "title generation",
    "ethics classification",
    "data to text",
    "answer verification",
    "wrong candidate generation",
    "cause effect classification",
    "stereotype detection",
    "answerability classification",
    "sentence ordering",
    "keyword tagging",
    "question understanding",
    "coherence classification",
    "sentence compression",
    "negotiation strategy detection",
    "spam classification",
    "dialogue act recognition",
    "section classification",
]

DOMAINS_TO_EXCLUDE = {
    "linguistics",
    "english exams",
}


class Args(Tap):
    output_dir: str = "path/to/dataset"
    seed: int = 42

    def process_args(self):
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)


# --- conversion ----------------------------------------------------------


def convert_to_sharegpt(example: dict, messages: list[dict]) -> dict | None:
    conversations = []
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        mapped_role = ROLE_MAP.get(role)
        if mapped_role is None:
            print(f"[WARN] Unknown role '{role}' in id={example['id']} — skipping turn")
            continue
        conversations.append({"from": mapped_role, "value": content})

    if not conversations:
        return None

    return {
        "id": example["id"],
        "source": example["source"],
        "conversations": conversations,
    }


def convert_and_save(examples: list[dict], output_path: Path) -> None:
    converted, skipped = [], 0
    for example in tqdm(examples, desc=f"> Converting {output_path.name}"):
        result = convert_to_sharegpt(example, example["messages"])
        if result is None:
            skipped += 1
        else:
            converted.append(result)

    print(f"  Converted : {len(converted):,}")
    print(f"  Skipped   : {skipped:,}")

    with open(output_path, "w", encoding="utf-8") as f:
        for record in converted:
            f.write(ujson.dumps(record, ensure_ascii=False))
            f.write("\n")

    print(f"  Saved to  : {output_path}")


# --- composition ---------------------------------------------------------


def stratified_split(
    examples: list[dict], fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split examples into (sampled, remainder) stratified by source.
    For each source, samples up to `fraction` of the available examples.
    If a source has fewer examples than the fraction would yield, all are sampled.
    """
    # rng = np.random.default_rng(seed)
    by_source: dict[str, list[dict]] = {}
    for ex in examples:
        by_source.setdefault(ex["source"], []).append(ex)

    sampled, remainder = [], []
    for group in by_source.values():
        rng = np.random.default_rng(seed)
        n = min(len(group), max(1, round(len(group) * fraction)))
        indices = rng.choice(len(group), size=n, replace=False)
        idx_set = set(indices.tolist())
        for i, ex in enumerate(group):
            (sampled if i in idx_set else remainder).append(ex)

    return sampled, remainder


def with_en(ex: dict) -> dict:
    return {**ex, "messages": ex["en_messages"]}


def with_it(ex: dict) -> dict:
    return {**ex, "messages": ex["it_messages"]}


def compose_datasets(
    t3_base: list[dict],
    t3_to_sample: list[dict],
    seed: int,
) -> dict[str, list[dict]]:
    """
    t3_base      : always-included samples (untranslatable EN + multilingual),
                   already have a `messages` key (messages_orig)
    t3_to_sample : translatable samples with both `en_messages` and `it_messages`

    control/sub_ variants have size len(t3_base) + len(t3_to_sample).
    add_ variants are larger by len(t3_it_half) or len(t3_to_sample).
    """
    t3_it_half, t3_it_remainder = stratified_split(t3_to_sample, 0.5, seed)

    return {
        "dromedario__control": [with_en(ex) for ex in t3_to_sample] + t3_base,
        "dromedario__sub_50": [with_it(ex) for ex in t3_it_half]
        + [with_en(ex) for ex in t3_it_remainder]
        + t3_base,
        "dromedario__sub_100": [with_it(ex) for ex in t3_to_sample] + t3_base,
        "dromedario__add_50": [with_en(ex) for ex in t3_to_sample]
        + t3_base
        + [with_it(ex) for ex in t3_it_half],
        "dromedario__add_100": [with_en(ex) for ex in t3_to_sample]
        + t3_base
        + [with_it(ex) for ex in t3_to_sample],
    }


# --- main ----------------------------------------------------------------


def main(args: Args):
    print("Loading T3-it...")
    t3_it_full = load_dataset(TULU3_IT_NAME, split="train")
    # schema
    # id, source, task, domain, lang_orig, messages_orig, messages_transl, flags

    # derive partitions from the annotation layer, excluding unwanted sources
    multilingual_ids = set()
    untranslatable_ids = set()
    translatable_ids = set()
    for ex in t3_it_full:
        if ex["source"] in EXCLUDED_SOURCES:
            continue
        if ex["lang_orig"] != "en":
            multilingual_ids.add(ex["id"])
        elif (
            ex["messages_transl"] is None
            or ex["domain"].lower() in DOMAINS_TO_EXCLUDE
            or ex["task"].lower() not in TASKS_TO_KEEP
        ):
            untranslatable_ids.add(ex["id"])
        else:
            translatable_ids.add(ex["id"])

    always_include_ids = multilingual_ids | untranslatable_ids

    # base: multilingual + untranslatable EN, using original messages
    t3_base = list(
        t3_it_full.filter(lambda x: x["id"] in always_include_ids)
        .map(lambda x: {"messages": x["messages_orig"]})
        .remove_columns(["messages_transl"])
    )

    # translatable pool: keep both EN and IT messages for composition
    t3_to_sample = list(
        t3_it_full.filter(lambda x: x["id"] in translatable_ids).map(
            lambda x: {
                "en_messages": x["messages_orig"],
                "it_messages": x["messages_transl"],
            }
        )
    )

    print(f"  T3 (excl. hard-coded)             : {len(t3_base) + len(t3_to_sample):,}")
    print(f"  T3-it (translatable)              : {len(t3_to_sample):,}")
    print(f"  T3-base (untransl. + multilingual): {len(t3_base):,}")
    print(f"    of which multilingual           : {len(multilingual_ids):,}")
    print(f"    of which untranslatable         : {len(untranslatable_ids):,}")
    # quit()

    compositions = compose_datasets(t3_base, t3_to_sample, args.seed)

    for name, examples in compositions.items():
        print(f"\nProcessing '{name}' ({len(examples):,} examples)...")
        filename = f"tulu3___{name}___sharegpt.jsonl"
        output_path = args.output_path / filename
        if output_path.exists():
            print(f"  Skipping  : {output_path} already exists")
            continue
        convert_and_save(examples, output_path)

    print("Conversion done.")
    print(f"Copy config/dataset_info.json into {args.output_path} (the same directory you will")
    print("set as `dataset_dir` in config/data/*.yaml) -- LLaMA-Factory reads dataset entries from")
    print("that file, and its keys already match the `dataset:` fields in config/data/*.yaml.")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
