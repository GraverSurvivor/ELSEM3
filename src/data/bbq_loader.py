"""Loads the BBQ benchmark, ambiguous condition only, for the project's 5 categories.

Source dataset: https://huggingface.co/datasets/heegyu/bbq (one config per bias category,
mirrors the original BBQ release https://arxiv.org/abs/2110.08193).

Loaded via the `refs/convert/parquet` revision rather than the dataset's own loading script:
`datasets>=4.0` refuses to execute third-party dataset scripts at all (security change), and
the repo's own `heegyu/bbq` script errors under that version with "Dataset scripts are no
longer supported". The auto-converted Parquet mirror has the identical schema and works
regardless of `datasets` version, so we use it unconditionally instead of branching on version.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, load_dataset

# Project category name -> HF config name on heegyu/bbq.
CATEGORIES: dict[str, str] = {
    "Age": "Age",
    "Gender_identity": "Gender_identity",
    "Race_ethnicity": "Race_ethnicity",
    "Religion": "Religion",
    "SES": "SES",
}

DATASET_ID = "heegyu/bbq"


@dataclass(frozen=True)
class BBQItem:
    """One ambiguous-condition BBQ question, normalized for the eval harness."""

    example_id: int
    category: str
    question_polarity: str
    context: str
    question: str
    options: tuple[str, str, str]
    label: int
    unknown_index: int

    @property
    def question_id(self) -> str:
        """Stable ID for safe-set / flipped-set tracking: unique per category+example."""
        return f"{self.category}:{self.example_id}"


def _find_unknown_index(answer_info: dict[str, list[str]]) -> int:
    """Returns which of ans0/ans1/ans2 is the "unknown"/"cannot be determined" option.

    BBQ's answer_info maps each ansN to [answer_text, group_label]; the unknown option's
    group_label is always "unknown" (case-insensitive) in the source dataset.
    """
    for i, key in enumerate(("ans0", "ans1", "ans2")):
        if "unknown" in answer_info[key][1].lower():
            return i
    raise ValueError(f"No unknown-labeled option found in answer_info: {answer_info}")


def _to_item(row: dict, category: str) -> BBQItem:
    unknown_index = _find_unknown_index(row["answer_info"])
    return BBQItem(
        example_id=row["example_id"],
        category=category,
        question_polarity=row["question_polarity"],
        context=row["context"],
        question=row["question"],
        options=(row["ans0"], row["ans1"], row["ans2"]),
        label=row["label"],
        unknown_index=unknown_index,
    )


def load_bbq_ambiguous(category: str) -> list[BBQItem]:
    """Loads one category's ambiguous-condition rows as normalized BBQItems.

    `category` must be a key of CATEGORIES (project naming), e.g. "SES", not "Socioeconomic".
    """
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; expected one of {list(CATEGORIES)}")

    config_name = CATEGORIES[category]
    ds: Dataset = load_dataset(
        DATASET_ID, split="test", revision="refs/convert/parquet", data_dir=config_name
    )
    ambig = ds.filter(lambda row: row["context_condition"] == "ambig")
    return [_to_item(row, category) for row in ambig]


def load_all_categories() -> dict[str, list[BBQItem]]:
    """Loads ambiguous-condition items for all 5 project categories, keyed by category name."""
    return {category: load_bbq_ambiguous(category) for category in CATEGORIES}


def summarize_splits(items_by_category: dict[str, list[BBQItem]]) -> dict[str, int]:
    """Row counts per category plus a "total" key — used to sanity-check the loader."""
    counts = {category: len(items) for category, items in items_by_category.items()}
    counts["total"] = sum(counts.values())
    return counts
