"""Offline checks for scripts/python/slides/build_deck.py against the real ASU template."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

pptx = pytest.importorskip("pptx")
from pptx.shapes.placeholder import PicturePlaceholder, PlaceholderPicture  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "python" / "slides"
SCRIPT = SCRIPTS / "build_deck.py"


@pytest.fixture(scope="module")
def build_deck_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("build_deck", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_deck"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def template(build_deck_module) -> Path:
    try:
        return build_deck_module.find_default_template()
    except FileNotFoundError:
        pytest.skip("ASU template not present in docs/")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


IMAGE_SIZES = {"find_query.png": (850, 672), "find_results.png": (843, 845),
               "chat.png": (837, 799), "taxonomy.png": (1702, 1305)}


def _write_images(folder: Path) -> Path:
    from PIL import Image

    folder.mkdir()
    for name, size in IMAGE_SIZES.items():
        Image.new("RGB", size, (200, 120, 40)).save(folder / name)
    return folder


@pytest.fixture
def built(build_deck_module, template, tmp_path):
    before = _sha256(template)
    images = _write_images(tmp_path / "images")
    out = build_deck_module.build_deck(template=template, out=tmp_path / "deck.pptx",
                                       images_dir=images)
    return out, template, before


def _slide_text(slide) -> str:
    parts = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            parts.append(shape.text_frame.text)
        elif shape.has_table:
            parts += [cell.text_frame.text for row in shape.table.rows for cell in row.cells]
    return "\n".join(parts)


def test_cover_slide_text(built):
    prs = pptx.Presentation(str(built[0]))
    assert len(prs.slides) == 10
    texts = [s.text_frame.text for s in prs.slides[0].placeholders]
    assert "GetWork: Two-Way Resume and Job Matching with a Skill Taxonomy" in texts
    assert "CSE 573 · Group 6 · Project 3" in texts
    assert "Omar Abuasba and Prathyush Vasa" in texts
    disclosure = _slide_text(prs.slides[0])
    assert "AI disclosure: Claude (generative AI" in disclosure


UI_SLIDE_INDEXES = (7, 8)  # screenshot slides; a slot is empty only when its file is absent


def _is_empty_picture_slot(shape) -> bool:
    return isinstance(shape, PicturePlaceholder)


def _filled_pictures(slide) -> list:
    return sorted((s for s in slide.shapes if isinstance(s, PlaceholderPicture)),
                  key=lambda s: s.left)


def test_no_empty_placeholders(built):
    """With all four image files present, no placeholder anywhere is left empty."""
    prs = pptx.Presentation(str(built[0]))
    for index, slide in enumerate(prs.slides):
        for shape in slide.placeholders:
            assert not _is_empty_picture_slot(shape), (index, shape.name)
            if not isinstance(shape, PlaceholderPicture):
                assert shape.text_frame.text.strip(), (index, shape.name)


def test_ui_slides_embed_pictures_with_aspect_preserved(built):
    prs = pptx.Presentation(str(built[0]))
    expected = {7: ["find_query.png", "find_results.png"], 8: ["chat.png", "taxonomy.png"]}
    for index, names in expected.items():
        pictures = _filled_pictures(prs.slides[index])
        assert len(pictures) == 2
        for picture, name in zip(pictures, names):
            width, height = IMAGE_SIZES[name]
            assert (picture.crop_left, picture.crop_right) == (0, 0)
            assert (picture.crop_top, picture.crop_bottom) == (0, 0)
            assert abs(picture.width / picture.height - width / height) < 0.01
        assert pictures[0].left + pictures[0].width <= pictures[1].left


def test_ui_captions_replace_old_screenshot_labels(built):
    prs = pptx.Presentation(str(built[0]))
    find, chat = _slide_text(prs.slides[7]), _slide_text(prs.slides[8])
    assert "Screenshot:" not in find + chat
    assert "Find jobs: paste a resume, fill in a form, or pick an example" in find
    assert "Top-10 postings, each with shared, related and missing skills" in find
    assert "Chat about a suggestion: ask why it matched or how to improve the resume" in chat
    assert "Skill clusters: 1,000 skills in 10 clusters (k-means, 2-D UMAP view)" in chat
    notes = prs.slides[8].notes_slide.notes_text_frame.text
    assert "separate 2-D UMAP" in notes


def test_missing_images_leave_empty_slots_and_warn(build_deck_module, template, tmp_path,
                                                   capsys):
    out = build_deck_module.build_deck(template=template, out=tmp_path / "d.pptx",
                                       images_dir=tmp_path / "nope")
    assert capsys.readouterr().err.count("warning: no screenshot") == 4
    prs = pptx.Presentation(str(out))
    for index in UI_SLIDE_INDEXES:
        empties = [p for p in prs.slides[index].placeholders if _is_empty_picture_slot(p)]
        assert len(empties) == 2


def test_taxonomy_falls_back_to_cluster_map(build_deck_module, template, tmp_path, capsys):
    images = _write_images(tmp_path / "images")
    (images / "taxonomy.png").rename(images / "cluster_map.png")
    out = build_deck_module.build_deck(template=template, out=tmp_path / "d.pptx",
                                       images_dir=images)
    assert "using cluster_map.png" in capsys.readouterr().err
    assert len(pptx.Presentation(str(out)).slides) == 10


def test_template_bytes_unchanged(built):
    _, template, before = built
    assert _sha256(template) == before


def test_no_orphaned_slide_parts(built):
    with zipfile.ZipFile(built[0]) as archive:
        slides = [n for n in archive.namelist() if n.startswith("ppt/slides/slide")]
    assert len(slides) == 10


def test_refuses_to_overwrite_template(build_deck_module, template):
    with pytest.raises(ValueError):
        build_deck_module.build_deck(template=template, out=template)


EXPECTED_SLIDE_TEXT = {
    1: ["Problem Definition", "Same skills, different vocabulary", "Requires React",
        "Built SPAs in Vue", "Research question:"],
    2: ["System Architecture", "Build once, then match per request",
        "Build phase (runs once on the corpus)", "Query phase (runs per request)",
        "1  Filter", "3  Select skills", "6  Embed docs", "7  Two-way matching", "8  Explain",
        "Classical (rules, statistics, non-neural ML)", "Modern (LLMs, neural embeddings)",
        "k-means, k=10", "Serving layer (web app)", "React 18 + Vite UI",
        "find, results, chat pages", "FastAPI + uvicorn API", "find, explain, chat",
        "chosen based on what was available on ASU Voyager"],
    3: ["Algorithms", "From raw text to a ranked list in four steps",
        "1  Structure and canonicalize", "up to 3 attempts", "JS becomes JavaScript",
        "2  Build the skill taxonomy", "Top 1,000 canonical skills",
        "UMAP (5 dimensions, 15 neighbors)", "29 configurations", "30% noise cap",
        "k-means, 10 clusters", "HDBSCAN scored higher, left skills unclustered",
        "3  Re-label and serialize", "Front-end Development: JavaScript, HTML, CSS",
        "partial credit", "from years of experience for resumes", "4  Retrieval", "L2-normalized", "S = R · J^T",
        "by row (jobs for a resume) or by column (candidates for a posting)",
        "default weights, untuned", "B1+ vs. B2 isolates embeddings, B2 vs. Full the taxonomy"],
    4: ["Data Set Description", "29,783", "~124K", "3+ yrs", "Jiechieu and Tsopze 2021"],
    5: ["Evaluation Plan", "Three systems, blind LLM judges, ranked-list metrics",
        "Preliminary evaluation plan", "B1+ (keyword, untuned default weights)",
        "30 resumes + 30 postings", "1,286 unique pairs judged", "Gemma 4 31B (Google)",
        "Llama 4 Maverick 17B (Meta)", "rubric judge_v1 frozen before the test run",
        "nDCG@10 (graded gain 0/1/2)", "paired randomization test",
        "Upcoming evaluation plan", "Skills retained", "Clusters", "Judge prompts",
        "Judge models", "Directions: do system gaps differ for resume to posting vs. posting to resume?",
        "Substitution test (not yet run): React to Vue vs. React to Java rank drop",
        "Explanation faithfulness (not yet run): judge rating 1-5, hallucination rate, hand review",
        "Human evaluation"],
    6: ["Preliminary Results", "[Headline: one sentence finding goes here]", "Differences",
        "B2 - B1+", "Full - B2", "Full - B1+", "0.000 [0.000, 0.000]", "+0.000", "p = 0.000",
        "Judge agreement", "Exact agreement: [00%]",
        "Cohen's kappa: [0.00] (weighted [0.00])", "Pairs judged: [N]", "Takeaways",
        "[Finding 1]", "[Finding 2]", "[Caveat: ...]",
        "the full evaluation plan has not been run yet",
        "Not yet reported: substitution test, explanation faithfulness, human check."],
    7: ["UI Interfaces", "Find matches in both directions",
        "Submit a resume to find jobs, or a posting to find candidates",
        "Each result shows a similarity score and which skills match, relate or are missing",
        "\"Related\" skills earn partial credit through shared skill clusters"],
    9: ["Project Plan", "Done", "Upcoming", "Parameter tuning", "Human evaluation",
        "Full evaluation run", "Final report and demo", "Division of work",
        "Both authors contribute to every stage, working jointly within each week",
        "Both label the human set (about 7 hours each) and review explanations",
        "Shared repository; weekly progress review to adjust the timeline",
        "GitHub: <repo link>"],
    8: ["UI Interfaces and Data Visualization", "Chat explanations and the skill taxonomy",
        "The assistant answers follow-up questions about the matched posting",
        "It warns that it is an AI and can be wrong, and points users to the skills on the card",
        "Clusters group related skills, such as Front-end, Operating Systems and Database"],
}


@pytest.mark.parametrize("index", sorted(EXPECTED_SLIDE_TEXT))
def test_content_slide_key_text(built, index):
    slide = pptx.Presentation(str(built[0])).slides[index]
    text = _slide_text(slide)
    for fragment in EXPECTED_SLIDE_TEXT[index]:
        assert fragment in text


def test_eval_plan_slide_merges_old_boxes(built):
    slide = pptx.Presentation(str(built[0])).slides[5]
    titles = [s.text_frame.paragraphs[0].text for s in slide.shapes if s.has_text_frame]
    assert titles.count("Preliminary evaluation plan") == 1
    for gone in ("Judge agreement", "Setup", "Metrics"):
        assert gone not in titles
    assert "Judge agreement" not in _slide_text(slide)
    plan = next(s for s in slide.shapes if s.has_text_frame
                and s.text_frame.paragraphs[0].text == "Preliminary evaluation plan")
    assert "nDCG@10 (graded gain 0/1/2), P@10, MRR" in plan.text_frame.text


def _bold_text(slide) -> set[str]:
    return {run.text for shape in slide.shapes if shape.has_text_frame
            for paragraph in shape.text_frame.paragraphs for run in paragraph.runs
            if run.font.bold}


def test_algorithm_names_and_structured_counts_are_bold(built):
    prs = pptx.Presentation(str(built[0]))
    algorithms = _bold_text(prs.slides[3])
    for name in ("UMAP", "k-means", "DBSCAN", "HDBSCAN", "silhouette", "cosine", "Jaccard",
                 "L2-normalized", "alias table", "LLM"):
        assert name in algorithms
    datasets = _bold_text(prs.slides[4])
    assert {"10,567", "8,416"} <= datasets
    assert "29,783" in datasets and "28,842" not in datasets and "8,423" not in datasets


def test_direction_takeaway_from_real_run_numbers(build_deck_module, template, tmp_path):
    out = build_deck_module.build_deck(
        template=template, out=tmp_path / "d.pptx",
        eval_dir=REPO_ROOT / "eval" / "preliminary_results")
    results = _slide_text(pptx.Presentation(str(out)).slides[6])
    assert "Posting to resume: B2 over B1+ +0.195 (vs. +0.117); Full vs. B2 unclear in both" in results


def test_text_fit_estimator_counts_wrapping_and_paragraph_spacing():
    import check_deck_layout
    from pptx.util import Emu, Inches, Pt

    prs = pptx.Presentation()
    blank = prs.slides.add_slide(prs.slide_layouts[6])
    box = blank.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(0.4))
    frame = box.text_frame
    frame.word_wrap = True
    frame.text = "word " * 40
    frame.paragraphs[0].runs[0].font.size = Pt(11)
    tall = check_deck_layout.estimated_text_height_emu(box)
    assert tall > Inches(1.5)
    assert any("text overflow" in p for p in check_deck_layout.check_slide(blank))
    box.height = Emu(int(tall) + 1)
    frame.paragraphs[0].space_before = Pt(10)
    assert check_deck_layout.estimated_text_height_emu(box) > tall


def test_all_shapes_within_slide_bounds_and_fit(built):
    import check_deck_layout

    prs = pptx.Presentation(str(built[0]))
    for number, slide in enumerate(prs.slides, start=1):
        assert check_deck_layout.check_slide(slide) == [], f"slide {number}"


def test_text_contrast_meets_wcag_aa(built):
    import check_deck_contrast

    prs = pptx.Presentation(str(built[0]))
    theme, cmap = check_deck_contrast.theme_colors(prs), check_deck_contrast.color_map(prs)
    for number, slide in enumerate(prs.slides, start=1):
        for owner, run, checker in check_deck_contrast._runs(slide):
            if not run.text.strip():
                continue
            text, back, ratio = checker(owner, run, theme=theme, cmap=cmap)
            assert ratio >= check_deck_contrast.MIN_CONTRAST, (number, run.text, text, back)


def test_template_color_map_swaps_background_two_to_dark(built):
    """Root cause of the earlier dark-gray cards: bg2 maps to dk2, a mid-dark gray."""
    import check_deck_contrast

    prs = pptx.Presentation(str(built[0]))
    resolved = check_deck_contrast.resolve_slot(
        "bg2", check_deck_contrast.theme_colors(prs), check_deck_contrast.color_map(prs))
    assert check_deck_contrast.luminance(resolved) < 0.3


def _write_eval_dir(path: Path) -> Path:
    system = {"mean": 0.5, "ci": [0.4, 0.6]}
    direction = {
        "n_queries": 40,
        "systems": {s: {m: dict(system) for m in ("P@5", "P@10", "MRR", "nDCG@10")}
                    for s in ("B1+", "B2", "Full")},
        "gaps": {g: {m: {"diff": 0.0123, "ci": [0.0, 0.02], "p": 0.04}
                     for m in ("P@5", "P@10", "MRR", "nDCG@10")}
                 for g in ("B2-B1+", "Full-B2", "Full-B1+")},
    }
    posting_direction = {**direction, "n_queries": 55}
    metrics = {"schema": 1, "groups": {"judge@test": {"directions": {
        "resume_to_posting": direction, "posting_to_resume": posting_direction}}}}
    agreement = {"schema": 1, "judges": {"test": {
        "n": 321, "pct_agree": 0.8, "kappa": 0.612, "kappa_quadratic": 0.7, "confusion": []}}}
    path.mkdir()
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (path / "agreement.json").write_text(json.dumps(agreement), encoding="utf-8")
    return path


def test_eval_dir_fills_slides_6_and_7(build_deck_module, template, tmp_path):
    eval_dir = _write_eval_dir(tmp_path / "eval")
    out = build_deck_module.build_deck(template=template, out=tmp_path / "d.pptx",
                                       eval_dir=eval_dir)
    prs = pptx.Presentation(str(out))
    design, results = _slide_text(prs.slides[5]), _slide_text(prs.slides[6])
    assert "Upcoming evaluation plan" in design
    assert "0.500 [0.400, 0.600]" in results
    assert "+0.012" in results and "p = 0.040" in results
    assert "Exact agreement: 80%" in results
    assert "Cohen's kappa: 0.61 (weighted 0.70)" in results
    assert "Pairs judged: 321" in results
    assert "[Finding 1]" not in results
    assert "Takeaways" in results and "(+0.012, p = 0.040)" in results
    assert "Judge kappa 0.61 is at or above the 0.6 target" in results
    assert "Posting to resume: B2 over B1+ +0.012 (vs. +0.012); Full vs. B2 same in both" in results
    assert "Directions: " in design
    import check_deck_layout

    for index in (5, 6):
        assert check_deck_layout.check_slide(prs.slides[index]) == []


def test_missing_eval_files_warn_and_keep_placeholders(build_deck_module, template, tmp_path,
                                                       capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = build_deck_module.build_deck(template=template, out=tmp_path / "d.pptx",
                                       eval_dir=empty)
    assert "warning:" in capsys.readouterr().err
    prs = pptx.Presentation(str(out))
    assert "Preliminary evaluation plan" in _slide_text(prs.slides[5])
    assert "0.000 [0.000, 0.000]" in _slide_text(prs.slides[6])


def test_partial_eval_json_never_crashes(build_deck_module, template, tmp_path):
    eval_dir = tmp_path / "bad"
    eval_dir.mkdir()
    (eval_dir / "metrics.json").write_text('{"groups": {"judge@test": {}}}', encoding="utf-8")
    (eval_dir / "agreement.json").write_text("not json", encoding="utf-8")
    out = build_deck_module.build_deck(template=template, out=tmp_path / "d.pptx",
                                       eval_dir=eval_dir)
    assert "Pairs judged: [N]" in _slide_text(pptx.Presentation(str(out)).slides[6])


def test_architecture_notes_hold_footer_and_serving_boxes_are_neutral(built):
    prs = pptx.Presentation(str(built[0]))
    slide = prs.slides[2]
    notes = slide.notes_slide.notes_text_frame.text
    assert "New documents pass through steps 2, 5 and 6 before matching." in notes
    serving = [s for s in slide.shapes if s.has_text_frame
               and s.text_frame.text.startswith(("React 18", "FastAPI"))]
    assert len(serving) == 2
    for shape in serving:
        assert str(shape.fill.fore_color.rgb) == "F4F1F2"
