# Scripts

- `bash/`: pipeline and evaluation drivers (`run_pipeline.sh`, `run_evaluation.sh`,
  `run_evaluation_preliminary.sh`, `match_random.sh`). Run from anywhere as
  `bash scripts/bash/<name>.sh`; each script changes to the repo root itself.
- `python/slides/`: the presentation generator and its checks. Build with
  `uv run python scripts/python/slides/build_deck.py --eval-dir eval/preliminary_results`;
  check with `check_deck_layout.py` and `check_deck_contrast.py` in the same folder.
  `crop_taxonomy_image.py` crops the taxonomy screenshot.
- `python/other/`: `peek_dev_metrics.py`, a read-only look at dev-stage judge metrics.
