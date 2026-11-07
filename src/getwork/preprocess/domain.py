import re
from collections.abc import Sequence

import pandas as pd

from getwork.preprocess import filters

DOMAIN_FAMILIES = {  # insertion order is the family order
    "software": r"software|developer|programmer|full[- ]?stack|front[- ]?end|back[- ]?end|devops|site reliability|\bsre\b|\bsdet\b|qa engineer|data engineer|machine learning engineer|(ios|android|mobile|web|cloud) engineer",
    "database": r"database (admin|administrator|engineer|developer|architect)|\bdba\b",
    "sysadmin": r"systems? (admin|administrator|engineer)|sysadmin|it support|help ?desk|desktop support|infrastructure engineer|cloud architect",
    "network": r"network (admin|administrator|engineer|architect|technician|analyst)",
    "security": r"security (analyst|engineer|architect)|cyber|infosec|information security|soc analyst",
    "project_manager": r"project manager|program manager|scrum master",
}
TITLE_EXCLUDE = r"real estate|business developer|business development|property developer|land developer|\bsales\b|account executive|\brecruit|talent acquisition|hvac|merchandis|\bnurs"
PM_TECH_TITLE = r"\bIT\b|software|technical|technology|systems?|digital|data|application|\bERP\b|\bSAP\b|cloud|network|infrastructure|cyber|security|\bweb\b|agile|devops|\bAI\b"
TECH_INDUSTRIES = frozenset({
    "Software Development", "IT Services and IT Consulting", "IT System Custom Software Development",
    "Information Technology & Services", "Technology, Information and Internet",
    "Technology, Information and Media", "Computer and Network Security",
    "Data Security Software Products", "Desktop Computing Software Products",
    "Data Infrastructure and Analytics", "Computer Networking", "Computer Networking Products",
    "Computer Hardware", "Computer Hardware Manufacturing", "Telecommunications",
    "Internet Marketplace Platforms", "Social Networking Platforms", "Computer Games",
})
PM_FAMILY: str = "project_manager"
NOT_IN_DOMAIN = "not_in_domain"

_FAMILY_RES = {name: re.compile(p, re.IGNORECASE) for name, p in DOMAIN_FAMILIES.items()}
_EXCLUDE_RE = re.compile(TITLE_EXCLUDE, re.IGNORECASE)
_PM_TECH_RE = re.compile(PM_TECH_TITLE, re.IGNORECASE)


def domain_families(title: str, industries: Sequence[str]) -> list[str]:
    """Families whose pattern matches title, in DOMAIN_FAMILIES order.
    Return [] if TITLE_EXCLUDE matches the title.
    The PM_FAMILY is returned only if PM_TECH_TITLE matches the title
    or at least one industry is in TECH_INDUSTRIES. All other families
    depend on the title alone."""
    if _EXCLUDE_RE.search(title):
        return []
    families = [name for name, rx in _FAMILY_RES.items() if rx.search(title)]
    has_it_signal = bool(_PM_TECH_RE.search(title)) or any(
        ind in TECH_INDUSTRIES for ind in industries
    )
    if not has_it_signal and PM_FAMILY in families:
        families.remove(PM_FAMILY)
    return families


def apply_domain_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Input columns: id (str), title (str), industries (list[str]).
    Return (kept, dropped).
    kept: rows with at least one family, all input columns plus
          families (list[str]), same row order.
    dropped: columns id (str), rule (str) = "not_in_domain", detail (str) = ""."""
    families = [
        domain_families(title=t, industries=i)
        for t, i in zip(df["title"], df["industries"])
    ]
    is_kept = pd.Series([bool(f) for f in families], index=df.index)
    kept = df[is_kept].copy()
    kept["families"] = [f for f in families if f]
    dropped = pd.DataFrame(
        {"id": df.loc[~is_kept, "id"].astype(str).to_numpy()}
    )
    dropped["rule"] = NOT_IN_DOMAIN
    dropped["detail"] = ""
    return kept, dropped[filters.DROPPED_COLUMNS]
