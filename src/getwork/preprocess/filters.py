from dataclasses import dataclass

import pandas as pd

from getwork.preprocess.text import alpha_ratio, control_ratio, detect_language

DROPPED_COLUMNS = ["id", "rule", "detail"]


@dataclass(frozen=True)
class ResumeFilterConfig:
    min_words: int = 50
    max_control_ratio: float = 0.01
    min_alpha_ratio: float = 0.5
    non_english_min_prob: float = 0.90
    lang_sample_chars: int = 2000


def _first_failure(row: pd.Series, cfg: ResumeFilterConfig) -> tuple[str, str] | None:
    """Return (rule, detail) for the first rule the row fails, else None."""
    if row["n_words"] < cfg.min_words:
        return "too_short", f"n_words={row['n_words']}"

    ctrl = control_ratio(row["text_raw"])
    alpha = alpha_ratio(row["text_clean"])
    if ctrl > cfg.max_control_ratio or alpha < cfg.min_alpha_ratio:
        return "corrupted", f"control_ratio={ctrl:.4f} alpha_ratio={alpha:.4f}"

    lang, prob = detect_language(row["text_clean"], cfg.lang_sample_chars)
    if lang != "en" and prob >= cfg.non_english_min_prob:
        return "non_english", f"lang={lang} prob={prob:.2f}"
    return None


def apply_quality_filters(
    df: pd.DataFrame, cfg: ResumeFilterConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, dropped); dropped has columns id, rule, detail.

    Rules run in order too_short, corrupted, non_english; the first failure wins.
    The language detector only runs on rows that passed the first two rules.
    """
    failures = [_first_failure(row, cfg) for _, row in df.iterrows()]
    is_kept = pd.Series([f is None for f in failures], index=df.index, dtype=bool)
    dropped_rows = [
        {"id": row_id, "rule": f[0], "detail": f[1]}
        for row_id, f in zip(df["id"], failures)
        if f is not None
    ]
    dropped = pd.DataFrame(dropped_rows, columns=DROPPED_COLUMNS)
    return df[is_kept], dropped
