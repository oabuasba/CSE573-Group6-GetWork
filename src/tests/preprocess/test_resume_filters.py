import pandas as pd

from getwork.preprocess.filters import ResumeFilterConfig, apply_quality_filters

CFG = ResumeFilterConfig()
ENGLISH = (
    "I am a software engineer with many years of experience building web services. "
    "I have worked on databases, APIs, and deployment pipelines for large teams. "
    "My responsibilities included designing systems, reviewing code, and mentoring "
    "junior developers across several projects in the company over time. "
    "I enjoy learning new tools and working closely with product managers."
)
SPANISH = (
    "Soy un ingeniero de software con muchos años de experiencia en el desarrollo de "
    "aplicaciones web. He trabajado con bases de datos y servicios para equipos "
    "grandes en distintas empresas. Mis responsabilidades incluyeron el diseño de "
    "sistemas, la revisión de código y la tutoría de desarrolladores nuevos en "
    "varios proyectos durante mucho tiempo."
)


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """rows: (id, text_raw, text_clean)."""
    return pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "text_raw": [r[1] for r in rows],
            "text_clean": [r[2] for r in rows],
            "n_words": [len(r[2].split()) for r in rows],
        }
    )


def _words(n: int) -> str:
    """First n words of the English sample, so the language rule keeps it."""
    return " ".join(ENGLISH.split()[:n])


def test_too_short_boundary():
    df = _frame([("a", _words(49), _words(49)), ("b", _words(50), _words(50))])
    kept, dropped = apply_quality_filters(df, CFG)
    assert dropped.to_dict("records") == [
        {"id": "a", "rule": "too_short", "detail": "n_words=49"}
    ]
    assert list(kept.id) == ["b"]


def test_corrupted_rules():
    ctrl_text = ENGLISH + "\x01" * 20
    alpha_text = " ".join(["1234567890"] * 60)
    df = _frame(
        [
            ("ctrl", ctrl_text, ENGLISH),
            ("alpha", alpha_text, alpha_text),
            ("ok", ENGLISH, ENGLISH),
        ]
    )
    kept, dropped = apply_quality_filters(df, CFG)
    assert dict(zip(dropped.id, dropped.rule)) == {"ctrl": "corrupted", "alpha": "corrupted"}
    assert dropped.detail.iloc[0].startswith("control_ratio=")
    assert list(kept.id) == ["ok"]


def test_non_english_rule():
    unknown = " ".join(["1234"] * 60)
    df = _frame([("es", SPANISH, SPANISH), ("en", ENGLISH, ENGLISH), ("unk", unknown, unknown)])
    cfg = ResumeFilterConfig(min_alpha_ratio=0.0)
    kept, dropped = apply_quality_filters(df, cfg)
    assert list(dropped.id) == ["es"]
    assert dropped.rule.iloc[0] == "non_english"
    assert dropped.detail.iloc[0].startswith("lang=es prob=")
    assert list(kept.id) == ["en", "unk"]


def test_first_rule_wins_and_partition():
    short_and_corrupt = "\x01" * 10
    corrupt_and_spanish = SPANISH + "\x01" * 30
    df = _frame(
        [
            ("z", ENGLISH, ENGLISH),
            ("y", short_and_corrupt, short_and_corrupt),
            ("x", corrupt_and_spanish, SPANISH),
            ("w", SPANISH, SPANISH),
            ("v", ENGLISH, ENGLISH),
        ]
    )
    kept, dropped = apply_quality_filters(df, CFG)
    rules = dict(zip(dropped.id, dropped.rule))
    assert rules == {"y": "too_short", "x": "corrupted", "w": "non_english"}
    assert len(dropped) == 3
    assert set(kept.id).isdisjoint(set(dropped.id))
    assert set(kept.id) | set(dropped.id) == set(df.id)
    assert list(kept.id) == ["z", "v"]
    assert list(kept.columns) == list(df.columns)
