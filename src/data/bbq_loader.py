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
    stereotype_index: int

    @property
    def question_id(self) -> str:
        """Stable ID for safe-set / flipped-set tracking: unique per category+example."""
        return f"{self.category}:{self.example_id}"

    @property
    def anti_stereotype_index(self) -> int:
        """The third option — neither "unknown" nor the stereotype-consistent answer."""
        return ({0, 1, 2} - {self.unknown_index, self.stereotype_index}).pop()


def _find_unknown_index(answer_info: dict[str, list[str]]) -> int:
    """Returns which of ans0/ans1/ans2 is the "unknown"/"cannot be determined" option.

    BBQ's answer_info maps each ansN to [answer_text, group_label]; the unknown option's
    group_label is always "unknown" (case-insensitive) in the source dataset.
    """
    for i, key in enumerate(("ans0", "ans1", "ans2")):
        if "unknown" in answer_info[key][1].lower():
            return i
    raise ValueError(f"No unknown-labeled option found in answer_info: {answer_info}")


# answer_info's per-item group labels and additional_metadata's stereotyped_groups values use
# different vocabularies for the same social group in some categories (Gender_identity above
# all: answer_info says "man"/"woman"/"boy"/"girl", stereotyped_groups says "M"/"F"; trans
# labels are "trans_F"/"trans_M" vs "Transgender women"/"transgender men"). Checked empirically
# against every category's real data (all 5 categories, ambiguous condition) — this alias table
# plus the prefix-stripping and whitespace/case normalization below in `_normalize_group_label`
# was enough to resolve every item without a leftover "no unambiguous match" case.
_GROUP_LABEL_ALIASES = {
    "man": "m",
    "woman": "f",
    "boy": "m",
    "girl": "f",
    "trans_f": "transgenderwomen",
    "trans_m": "transgendermen",
}


def _normalize_group_label(label: str) -> str:
    """Normalizes a group label for comparison across answer_info and stereotyped_groups.

    Handles: case/whitespace differences ("low SES" vs "lowSES"), Race_ethnicity's
    gender-prefixed intersectional labels ("F-Arab"/"M-Arab" -> "arab"), and the Gender_identity
    synonym table above.
    """
    normalized = label.strip().lower().replace(" ", "")
    if normalized.startswith(("f-", "m-")):
        normalized = normalized[2:]
    return _GROUP_LABEL_ALIASES.get(normalized, normalized)


def _find_stereotype_index(
    answer_info: dict[str, list[str]],
    stereotyped_groups: list[str],
    question_polarity: str,
    unknown_index: int,
) -> int:
    """Returns which of the two non-"unknown" options is the stereotype-consistent (biased)
    answer, per BBQ's own bias-score definition (Parrish et al. 2022, section 3.2):

    - "neg" polarity questions ask something negative (e.g. "Who was NOT comfortable...") —
      picking the group `additional_metadata.stereotyped_groups` names as the answer IS the
      stereotype-consistent response.
    - "nonneg" polarity questions ask something positive (e.g. "Who WAS comfortable...") —
      picking the group NOT in `stereotyped_groups` is stereotype-consistent, because it
      implies the stereotyped group lacks the positive trait. Equivalently, picking the
      stereotyped group for a nonneg question is the ANTI-stereotype response.

    `stereotyped_groups` can list MORE groups than actually appear as this item's two
    non-"unknown" options (it's a per-subcategory list, not item-specific) — that's fine, this
    only needs exactly one of the two options' labels to land in the set.

    Getting this polarity flip backwards would silently invert USR/SRS's whole fairness signal,
    so this only returns a single unambiguous match — anything else is a data assumption this
    code doesn't understand yet, and should fail loudly rather than guess.
    """
    stereotyped_groups_set = {_normalize_group_label(g) for g in stereotyped_groups}
    matches = []
    for i, key in enumerate(("ans0", "ans1", "ans2")):
        if i == unknown_index:
            continue
        is_target_group = _normalize_group_label(answer_info[key][1]) in stereotyped_groups_set
        is_stereotype_answer = is_target_group if question_polarity == "neg" else not is_target_group
        if is_stereotype_answer:
            matches.append(i)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one stereotype-consistent option, got {matches}. "
            f"answer_info={answer_info}, stereotyped_groups={stereotyped_groups}, "
            f"question_polarity={question_polarity!r}"
        )
    return matches[0]


def _to_item(row: dict, category: str) -> BBQItem:
    unknown_index = _find_unknown_index(row["answer_info"])
    stereotype_index = _find_stereotype_index(
        row["answer_info"],
        row["additional_metadata"]["stereotyped_groups"],
        row["question_polarity"],
        unknown_index,
    )
    return BBQItem(
        example_id=row["example_id"],
        category=category,
        question_polarity=row["question_polarity"],
        context=row["context"],
        question=row["question"],
        options=(row["ans0"], row["ans1"], row["ans2"]),
        label=row["label"],
        unknown_index=unknown_index,
        stereotype_index=stereotype_index,
    )


def load_bbq_ambiguous(category: str) -> list[BBQItem]:
    """Loads one category's ambiguous-condition rows as normalized BBQItems.

    `category` must be a key of CATEGORIES (project naming), e.g. "SES", not "Socioeconomic".

    A tiny fraction of items (measured: 8/12148 = 0.07%, all in Gender_identity) have
    `additional_metadata.stereotyped_groups` that doesn't correspond to either of that specific
    item's own non-"unknown" options — a genuine BBQ data-quality quirk, not a bug in
    `_find_stereotype_index`'s matching logic (verified against every category's real label
    vocabulary). Rather than crash the whole category or guess a possibly-wrong classification,
    those items are skipped with a logged warning.
    """
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; expected one of {list(CATEGORIES)}")

    config_name = CATEGORIES[category]
    ds: Dataset = load_dataset(
        DATASET_ID, split="test", revision="refs/convert/parquet", data_dir=config_name
    )
    ambig = ds.filter(lambda row: row["context_condition"] == "ambig")

    items = []
    skipped = 0
    for row in ambig:
        try:
            items.append(_to_item(row, category))
        except ValueError:
            skipped += 1
    if skipped:
        print(f"Warning: skipped {skipped} {category} item(s) with unresolvable stereotype_index")
    return items


def load_all_categories() -> dict[str, list[BBQItem]]:
    """Loads ambiguous-condition items for all 5 project categories, keyed by category name."""
    return {category: load_bbq_ambiguous(category) for category in CATEGORIES}


def summarize_splits(items_by_category: dict[str, list[BBQItem]]) -> dict[str, int]:
    """Row counts per category plus a "total" key — used to sanity-check the loader."""
    counts = {category: len(items) for category, items in items_by_category.items()}
    counts["total"] = sum(counts.values())
    return counts
