---
title: TrueTag
emoji: 🏷️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# TrueTag: honest product descriptions

A website for fashion sellers. Upload a clothing photo and get a product description that never
invents a garment type, color or fabric the photo doesn't support.

This is the deployable application of the research project *Extract, Abstain, Generate:
Decomposing the Origins of Attribute Hallucination in Grounded E-Commerce Product Descriptions*
(Team 10, RV University; guide Prof. Lokesh J K). Research code:
[github.com/Yashas-BR/grounded-product-description](https://github.com/Yashas-BR/grounded-product-description).

## How it works

1. **Extract:** SigLIP (`google/siglip-so400m-patch14-384`) scores the photo against 13 garment types,
   12 colors and 8 fabrics. It uses the same prompt ensembling as the paper.
2. **Abstain:** an attribute below its confidence threshold becomes `unknown`. The thresholds were
   frozen on the human-labelled dev split of the study (`config.json`).
3. **Generate:** `Qwen/Qwen2.5-1.5B-Instruct` writes 2–3 sentences from the attributes, using the
   paper's frozen prompts.
4. **Verify:** the paper's rule-based claim checker reads the text back. In Safe mode, a draft that
   states an unsupported attribute is replaced by the slot-based APG rendering, which is safe by
   construction.

Writing modes on the site:

| Mode | Paper condition | Behaviour |
|---|---|---|
| Safe (default) | B-Abs + verify | Fluent text; uncertain attributes are left out; auto-corrected if the writer slips |
| Strict | APG | The system, not the AI, inserts attribute words; shorter text |
| No abstention | B | Always uses the top guess; shown for comparison |

## Run on your computer (Windows)

```bat
pip install -r requirements.txt
run_local.bat
```

Then open http://127.0.0.1:7860.

- The first start downloads about 7 GB of models. On this machine they are already in `D:\hf_cache`.
- Loading takes about a minute. After that, each photo takes roughly 5–20 seconds on a CPU.
- The page shows "Ready · CPU" when the models are loaded.

**Offline use:** once the models are downloaded, the site runs with no internet at all.
`run_local.bat` sets `HF_HUB_OFFLINE=1`, so nothing is fetched at startup. Offline, the page falls
back from the Inter web font to the system font.

On macOS or Linux:

```bash
pip install -r requirements.txt
uvicorn main:app --app-dir app --port 7860
```

## Host it (public link)

The app is one Docker container listening on port 7860. It needs about 12 GB of RAM on a CPU.

### Hugging Face Spaces (free, recommended)

1. Create a new Space at https://huggingface.co/new-space. Choose **Docker** as the SDK and **CPU basic**
   (free, 16 GB RAM) as the hardware.
2. Push this folder to the Space's git repository:
   ```bash
   git init
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git add .
   git commit -m "TrueTag"
   git push space main
   ```
   The front matter at the top of this README configures the Space.
3. Wait for the build (about 10 minutes). The first start downloads the models. Your link is
   `https://<your-username>-<space-name>.hf.space`.

### Any other host (Render, Railway, a VPS)

```bash
docker build -t truetag .
docker run -p 7860:7860 -v truetag-models:/data/hf_cache truetag
```

The volume keeps downloaded models between restarts. The `PORT` environment variable is respected.

### Settings

| Environment variable | Default | Effect |
|---|---|---|
| `GHD_LLM` | `Qwen/Qwen2.5-1.5B-Instruct` | Writer model. On a GPU host, `Qwen/Qwen2.5-7B-Instruct` matches the paper. |
| `HF_HOME` | `D:\hf_cache` locally, `/data/hf_cache` in Docker | Where models are stored |
| `PORT` | `7860` | Port (Docker only) |

## API

- `GET /api/health` returns `{"ready": true, "device": "cpu", ...}`.
- `POST /api/describe` takes multipart form data with `image` (a file) and `mode` (`safe`, `strict`
  or `forced`). It returns the detected attributes with confidences, the description, and the
  verification result.
- Interactive docs are at `/api/docs`.

## Project layout

```
app/main.py         web server (FastAPI): website + API
app/engine.py       pipeline: extract -> abstain -> generate -> verify
app/groundedgen/    research modules reused unchanged (taxonomy, prompts, claim checker, APG)
static/             the website (HTML, CSS, JavaScript)
config.json         models, frozen thresholds, limits
Dockerfile          container for hosting
```

## Differences from the research setup

The site uses the paper's frozen prompts and thresholds, with these additions for the small CPU writer:

- **Two extra system-prompt sentences:** do not mention fabric, material, texture or quality when the
  material is unknown, and do not describe fit, sizing or care. Without them the 1.5B model wrote
  things like "crafted from an unidentifiable material".
- **One format example in Strict mode.** Without it the 1.5B model returned a list of slots
  ("T-shirt: T-shirt") instead of sentences.
- **Output checks:**
  - list-like output falls back to the safe template
  - in Safe mode, when material is unknown, sentences that talk about fabric are removed
  - a draft naming an unsupported attribute is rebuilt with slots

The research results in the paper were produced without these additions.

## Limits

- The site is built for single-garment photos of 13 apparel types. Other products get the closest
  garment type, or `unknown` when confidence is low.
- Fabric is rarely stated. Zero-shot models cannot read fabric reliably from a photo (about 50%
  accuracy in the study), so the threshold keeps it `unknown` unless the model is very sure.
- The research used 7B writers on a GPU. The site's 1.5B writer slips more often, and the verify step
  exists to catch those slips.
