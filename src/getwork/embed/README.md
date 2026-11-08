# Embedding client

`getwork.embed` turns text into unit-length float32 vectors for the skill-taxonomy and
document-embedding stages. It offers a local backend (Qwen3-Embedding-4B through
sentence-transformers) and a Voyager backend (OpenAI-compatible `/embeddings`), a deterministic
fake for tests, batching with out-of-memory backoff, a chunk cache, and a vector store
(`write_vectors`, `read_vectors`).

## Options

Added to a stage CLI by `add_embed_args`: `--embed-backend {local,voyager}`, `--embed-model`,
`--device` (auto, cpu, cuda, cuda:N), `--dtype`, `--embed-batch-size`, `--max-seq-length`, and
`--env-file`.

## Backend precedence

`--embed-backend` over `EMBED_BACKEND` (environment over env file) over the default `local`.
Local needs no key and defaults to `Qwen/Qwen3-Embedding-4B` (`EMBED_MODEL` or `--embed-model`
overrides). Voyager needs `VOYAGER_API_KEY`, `VOYAGER_BASE_URL` and a model id from
`VOYAGER_EMBED_MODEL` or `--embed-model`, with no default.

## Same-embedding rule

Vector sets from different backends, models or dimensions must not be combined.
`check_same_embedding` compares model, backend and dim and raises `EmbeddingMismatchError`
otherwise. The same model id served by local and Voyager is still refused.

## Memory warning

The local LLM server (local-llm-backend) and the local embedding model do not fit a 12 GB card
together. Stop the LLM server before embedding.

## User-run checklist

No agent runs these. Items marked inferred come from general knowledge, not a source read;
verify them.

### EU-1 Install the embed extra

Run `uv sync --extra embed` (installs torch and sentence-transformers, several GB). The RTX 5070
is a Blackwell card (compute capability 12.0); inferred: it needs a PyTorch build for CUDA 12.8
or newer. Then run `uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`.
Pass: prints True and CUDA 12.8 or higher. If False, install the CUDA wheel index PyTorch
documents for your platform before going on.

### EU-2 Free the GPU

Stop any local LLM server and check `nvidia-smi`. Pass: nearly all 12 GB free.

### EU-3 Run the preflight

Run `uv run embed-preflight`. The first run downloads Qwen3-Embedding-4B from Hugging Face
(inferred: about 8 GB, `~/.cache/huggingface`); that download is yours to run. Pass: JSON with
backend local, model Qwen/Qwen3-Embedding-4B, a dim (inferred: 2560), dtype float16,
norms_ok true, and related_cosine greater than unrelated_cosine.

### EU-4 Watch memory

Watch `nvidia-smi` during the first real embedding run. A log line "reducing batch size" means
the run is fine but slower; pass `--embed-batch-size 4` to start lower. Pass: no crash, peak
memory below 12 GB. Note the batch size that worked.

### EU-5 Voyager embeddings (optional)

Add `VOYAGER_EMBED_MODEL=<id>` to `.env` (the id from Voyager's model list), then run
`uv run embed-preflight --embed-backend voyager`. Pass: JSON with backend voyager; dim matches
the local model's if both serve the 4B model.

### EU-6 Choose one backend

Choose one backend and model for the whole dataset and note it. `check_same_embedding` refuses
mixed sets. Pass: a written decision.
