import pandas as pd

DUPLICATE_RULE = "duplicate"


def _dedup_key(text_clean: str) -> str:
    return " ".join(text_clean.casefold().split())


def _merge_labels(label_lists: list[list[str]]) -> list[str]:
    return list(dict.fromkeys(label for labels in label_lists for label in labels))


def drop_duplicates_merge_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the smallest id per normalized-text group, with labels merged in id order.

    Returns (kept, dropped); dropped has columns id, rule, detail.
    """
    keys = df["text_clean"].map(_dedup_key)
    kept_id_by_key: dict[str, str] = {}
    merged_by_key: dict[str, list[str]] = {}
    dropped_rows: list[dict[str, str]] = []

    ordered = df.assign(_key=keys).sort_values("id", kind="stable")
    for key, group in ordered.groupby("_key", sort=False):
        kept_id_by_key[key] = group["id"].iloc[0]
        merged_by_key[key] = _merge_labels(list(group["labels"]))
        for dropped_id in group["id"].iloc[1:]:
            dropped_rows.append(
                {
                    "id": dropped_id,
                    "rule": DUPLICATE_RULE,
                    "detail": f"duplicate_of={kept_id_by_key[key]}",
                }
            )

    is_kept = df["id"].to_numpy() == keys.map(kept_id_by_key).to_numpy()
    kept = df[is_kept].copy()
    kept_keys = keys[is_kept]
    kept["labels"] = pd.Series(
        [merged_by_key[key] for key in kept_keys], index=kept.index, dtype=object
    )
    kept["first_label"] = kept["labels"].map(lambda labels: labels[0] if labels else None)
    kept["has_label"] = kept["labels"].map(lambda labels: len(labels) > 0).astype(bool)

    dropped = pd.DataFrame(dropped_rows, columns=["id", "rule", "detail"])
    return kept, dropped
