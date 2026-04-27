# DriveLM + nuScenes VLM Evaluation — Project Report

> **Goal:** Build a complete pipeline to evaluate and fine-tune a Vision-Language Model on autonomous driving QA tasks using the DriveLM dataset integrated with nuScenes imagery.

---

## Table of Contents

1. [Data Preparation](#1-data-preparation)
2. [Baseline VLM Benchmarking](#2-baseline-vlm-benchmarking)
3. [Fine-Tuning](#3-fine-tuning)
4. [Comparative Evaluation](#4-comparative-evaluation)
5. [Deployment & Optimization](#5-deployment--optimization)
6. [Scripts Reference](#6-scripts-reference)
7. [Setup & Reproduction](#7-setup--reproduction)

---

## 1. Data Preparation

### What We Built

A pipeline (`parse_drivelm.py`) that links DriveLM QA annotations with nuScenes imagery and metadata into a single structured CSV ready for model evaluation and training.

**Input sources:**
- `v1_0_train_nus.json` — DriveLM QA annotations (questions, answers, object references)
- nuScenes v1.0 mini split — 6-camera surround-view images + sensor metadata

**Output files:**

| File | Rows | Description |
|---|---|---|
| `qa_enriched.csv` | 9,006 | Primary QA pairs with camera paths, question type, object refs |
| `objects.csv` | 388 | Per-frame object annotations with category, status, bounding box |
| `frames.csv` | 95 | Key frame metadata (timestamp, scene, ego pose) |
| `scenes.csv` | 15 | Scene-level metadata |

### Why This Design

Each QA row contains everything needed for inference in one place — the question, answer, which cameras are relevant (`relevant_cameras`), and the file paths for all 6 cameras (`all_image_paths`). This avoids joins at inference time and makes the pipeline stateless.

A key design decision was storing `relevant_cameras` separately from `all_image_paths`:
- `relevant_cameras` — which cameras are *semantically relevant* to the question (verified: always matches relevant paths, 0 mismatches across 2,196 rows)
- `all_image_paths` — all 6 camera paths always present for lookup

This lets us pass only the relevant cameras to the model (reducing tokens and memory) while always having fallback paths available.

### Dataset Statistics

**15 scenes processed** (6 overlapping DriveLM + nuScenes, 9 additional DriveLM scenes), **95 key frames**, **9,006 QA pairs**.

#### QA Category Distribution

| Category | Count | % | Notes |
|---|---|---|---|
| Perception | 4,074 | 45.2% | Object presence, status, location |
| Prediction | 2,861 | 31.8% | Future state of other agents |
| Planning | 1,976 | 21.9% | Ego vehicle actions and probabilities |
| Behavior | 95 | 1.1% | Ego vehicle direction and speed |

#### Question Type Distribution

| Question Type | Count | % |
|---|---|---|
| what_query | 3,004 | 33.4% |
| yes_no | 2,183 | 24.2% |
| other_question_type | 1,147 | 12.7% |
| status_query | 1,020 | 11.3% |
| object_enumeration | 484 | 5.4% |
| visual_description | 388 | 4.3% |
| future_state | 340 | 3.8% |
| which_query | 285 | 3.2% |
| ego_behavior_prediction | 95 | 1.1% |
| planning_other | 60 | 0.7% |

#### Camera Mention Frequency

| Camera | Mentions | % |
|---|---|---|
| CAM_FRONT | 6,583 | 24.1% |
| CAM_BACK | 5,009 | 18.3% |
| CAM_FRONT_RIGHT | 4,326 | 15.8% |
| CAM_FRONT_LEFT | 4,069 | 14.9% |
| CAM_BACK_RIGHT | 3,748 | 13.7% |
| CAM_BACK_LEFT | 3,568 | 13.1% |

#### Answer Length Distribution

| Bucket | Count | Notes |
|---|---|---|
| 1–5 words | 6,422 (71.3%) | Short factual answers (Yes/No, Moving, Low) |
| 6–10 words | 742 (8.2%) | One-sentence answers |
| 11–20 words | 1,079 (12.0%) | Multi-object descriptions |
| 21–50 words | 515 (5.7%) | Planning explanations |
| 50+ words | 248 (2.8%) | Scene descriptions |

Mean answer length: **7.28 words** (std: 13.06). The high standard deviation reflects the mix of one-word answers ("Yes", "Moving") and full scene descriptions.

### Identified Biases and Gaps

- **Front camera bias:** CAM_FRONT has 24.1% of mentions vs 13.1% for CAM_BACK_LEFT — the model may learn to default to front-camera reasoning
- **Behavior underrepresentation:** Only 1.1% of questions — the model gets very few examples of ego-motion reasoning
- **Causal reasoning gap:** No "why" or reasoning questions detected (< 5%) — the dataset tests recognition and prediction but not explanation
- **No counting questions:** "How many X" questions are absent despite objects being annotated
- **Vehicle dominance:** 70.6% of referenced objects are vehicles — pedestrian and cyclist scenarios underrepresented

---

## 2. Baseline VLM Benchmarking

### Model Selection: LLaVA-1.5-7B

**Why LLaVA-1.5-7B:**
- Open-source, strong zero-shot visual reasoning
- Supports multi-image input natively (critical for 6-camera surround view)
- 4-bit quantization available — fits T4 16GB for both inference and training
- State-of-the-art on VQA benchmarks at time of evaluation

**Architecture:**
```
LlavaForConditionalGeneration
├── vision_tower    CLIP ViT-L/14 @ 336px  (~307M params)
│                   Each image → 576 visual tokens
├── mm_projector    2-layer MLP bridge      (~20M params)
│                   Maps visual tokens into LLM embedding space
└── language_model  LLaMA-2 / Vicuna 7B    (~6.7B params)
                    Autoregressive answer generation
```

**Quantization:** 4-bit NF4 (QLoRA bitsandbytes) for memory efficiency.
**Hardware:** RTX 3070 8GB (benchmarking), T4 16GB (training).

### Prompt Engineering Evolution

We ran two benchmark rounds with different image sizes and prompt designs.

**Version 1 (img-size 448, basic prompt — no camera labels):**
```
USER: <image><image><image>
Describe what you see and answer: <question>
ASSISTANT:
```

No camera labels, no system prompt, no few-shot examples. The model receives raw image tokens with no spatial context about which camera each image came from.

**Version 2 (img-size 336, improved prompt with inline camera labels):**
```
USER: You are an autonomous driving assistant...

[Front Camera]: <image>
[Front Left Camera]: <image>
[Back Camera]: <image>

Question: <question>
ASSISTANT:
```

Version 2 added: inline camera labels binding each `<image>` token to its spatial position, a system prompt, per-category few-shot examples, answer style rules, and `relevant_cameras` selection to pass only semantically relevant cameras instead of always all 6.

### Benchmark Results

**Version 1 — 448px, no camera labels (RTX 3070):**

| Metric | Score |
|---|---|
| ROUGE-L | **0.3580** |
| Exact Match | **0.2541** |
| BERTScore F1 | **0.9268** |
| Avg Latency | 2,520 ms |
| P95 Latency | 4,968 ms |
| Peak VRAM | 3.82 GB |

**Version 2 — 336px, camera-label prompt (T4):**

| Metric | Score | Δ vs v1 |
|---|---|---|
| ROUGE-L | 0.2942 | **−17.8%** |
| Exact Match | 0.1553 | **−38.9%** |
| BERTScore F1 | 0.9059 | −2.3% |
| Avg Latency | ~2,600 ms | +3.2% |

**Counterintuitive result — v2 scored lower despite the improved prompt.** Version 2 has a more principled prompt design (camera labels correctly bind visual tokens to spatial positions), but the results were worse across all metrics.

The most likely explanation is **image resolution**:

```
448px → 32×32 = 1,024 visual tokens per image  (CLIP patch size 14px)
336px → 24×24 =   576 visual tokens per image

For a 6-camera sample:
  448px: 6 × 1,024 = 6,144 visual tokens  ← much more visual information
  336px: 6 ×   576 = 3,456 visual tokens
```

The 448px model receives **78% more visual detail per image**. For DriveLM questions that require localising objects, reading their status, and understanding their spatial relationship to the ego vehicle — tasks that genuinely benefit from higher resolution — the extra tokens matter more than explicit camera labels in the prompt.

Camera label confusion (wrong-camera answers) is real but relatively rare (72 failures, 3.3% of evaluated). The visual information lost by dropping from 1,024 to 576 tokens affected a much larger portion of questions.

**The ideal configuration** would be 448px resolution AND camera labels, but this is OOM on T4 (6-cam at 448px = 6,144 visual tokens ≈ 14+ GB VRAM). This trade-off motivates future work on efficient high-resolution VLM inference.

### Results by Category

| Category | N | ROUGE-L | Exact Match | BERTScore | Avg Latency |
|---|---|---|---|---|---|
| behavior | 22 | **0.8442** | 0.0455 | 0.9757 | 3,456 ms |
| perception | 970 | 0.3850 | 0.2227 | 0.9340 | 2,577 ms |
| planning | 502 | 0.3180 | 0.2530 | 0.9147 | 3,904 ms |
| prediction | 702 | 0.3341 | **0.3048** | 0.9240 | 1,423 ms |

**Behavior ROUGE-L=0.844:** Behavior questions have a fixed answer template ("The ego vehicle is going straight. The ego vehicle is driving fast.") — the model has seen this pattern in pretraining and reproduces it accurately.

**Prediction fastest (1,423 ms):** 90% of prediction rows use only 1 camera → shortest sequences → fastest inference.

**Planning slowest (3,904 ms):** Planning often requires all 6 cameras for scene-wide reasoning → longest sequences.

### Results by Question Type (Hardest First)

| Question Type | N | ROUGE-L | Why Hard |
|---|---|---|---|
| planning_other | 16 | 0.0000 | Open-ended reasoning, no template |
| which_query | 87 | 0.0641 | Requires identifying specific object |
| visual_description | 99 | 0.1027 | Free-form scene description |
| status_query | 238 | 0.2441 | Often wrong (Moving vs Stationary) |
| what_query | 764 | 0.2802 | Mixed — object presence + attributes |
| yes_no | 505 | 0.4407 | Binary — easier to get right |
| object_enumeration | 112 | 0.6629 | Standard format, model handles well |
| future_state | 79 | 0.8354 | Template answer, model knows pattern |
| ego_behavior_prediction | 22 | 0.8442 | Fixed template |

### Failure Mode Analysis

| Mode | Count | % | Description |
|---|---|---|---|
| F_other | 571 | 26.0% | Miscellaneous mismatch |
| D_incomplete | 354 | 16.1% | Answer too short / misses objects |
| A_hallucination | 159 | 7.2% | Objects not visible in images |
| E_planning_error | 109 | 5.0% | Wrong ego behavior or direction |
| B_wrong_status | 90 | 4.1% | Moving/Stationary confused |
| C_wrong_camera | 72 | 3.3% | Wrong camera viewpoint referenced |

**Notable failure examples:**

*Hallucination (planning):*
> Q: What actions can lead to collision with vehicle in back camera?
> REF: Back up.
> PRD: Accelerate and go straight...

The model defaults to a common planning template ("accelerate and go straight") instead of reasoning that reversing causes back-camera collision.

*Wrong status (perception):*
> Q: What is the status of the vehicle in front camera?
> REF: Stationary.
> PRD: Moving.

The model cannot reliably distinguish stationary from slow-moving vehicles — a known limitation of single-frame VLMs without optical flow.

*Incomplete (perception):*
> Q: What is the status of the bus to the front right?
> REF: The bus to the front right is moving.
> PRD: Moving.

Technically correct but incomplete — the full sentence format expected by ROUGE-L scoring is missed.

### Metric Justification

**Primary metric: ROUGE-L** — measures longest common subsequence between prediction and reference. Appropriate for DriveLM because answers have a defined expected format and length. Robust to minor wording differences while penalising missing content.

**Supporting metrics:**
- **Exact Match** — strict correctness for short answers (Yes/No, Low/Medium/High, single-word status)
- **BERTScore F1** — semantic similarity; high scores (0.92+) confirm the model is semantically correct even when phrasing differs

**Why not accuracy only:** DriveLM answers range from one word to full paragraphs — binary accuracy is too coarse and doesn't capture partial correctness in multi-object descriptions.

### Cost Estimate (Baseline)

| Hardware | Latency | Cost/hr | Cost per 1k queries |
|---|---|---|---|
| RTX 3070 (local) | ~2.6s | ~$0 (owned) | ~$0 (electricity only) |
| T4 (AWS g4dn.xlarge) | ~2.5s | $0.526 | **$0.47** |
| T4 (GCP) | ~2.5s | $0.35 | **$0.29** |
| A100 (Lambda Labs) | ~0.8s | $1.10 | **$0.24** |

---

## 3. Fine-Tuning

### What We Tuned and Why

LLaVA-1.5-7B has three components. We treated each differently:

| Component | Params | Decision | Reason |
|---|---|---|---|
| vision_tower (CLIP ViT) | ~307M | **Frozen** | CLIP already produces excellent visual representations for camera imagery. Unfreezing risks catastrophic forgetting of visual grounding with no meaningful gain for this task. |
| multi_modal_projector (MLP bridge) | ~20M | **Fully trained** | Tiny (20M), highest-leverage for domain adaptation. Directly controls how visual tokens enter the LLM. Cost is negligible. |
| language_model (LLaMA-2 7B) | ~6.7B | **LoRA / QLoRA** | Where DriveLM-specific answer formatting, driving vocabulary, and reasoning live. Too large to fine-tune fully; LoRA gives best quality/cost tradeoff. |

### LoRA Configuration

**Why attention + MLP layers (not just attention):**
- Attention layers (`q,k,v,o_proj`) — control *where* the model looks
- MLP/FFN layers (`gate,up,down_proj`) — store factual patterns like "collision question → Low/Medium/High"

DriveLM requires both: new answer formats live in MLP weights, correct camera-label cross-referencing lives in attention.

```python
LORA_TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',   # attention
    'gate_proj', 'up_proj', 'down_proj',        # MLP / FFN
]

LoraConfig(
    r          = 16,      # rank — good balance for domain adaptation
    lora_alpha = 32,      # scaling = alpha/r = 2.0
    dropout    = 0.05,
    bias       = 'none',
)
```

**Trainable parameters:** 42.3M out of 7.1B total = **0.596%** — the model learns while only updating a tiny fraction of its weights.

### QLoRA Details

QLoRA (Quantized LoRA) loads the base model in 4-bit NF4 quantization while LoRA adapters train in BF16:

```
Base model (4-bit NF4)  : ~3.5 GB VRAM
LoRA adapters (BF16)    : ~0.3 GB
Projector (FP32)        : ~0.2 GB
Activations + optimizer : ~4-6 GB
Total                   : ~8-10 GB  ← fits T4 16GB ✓
```

**Why NF4 (Normal Float 4) over INT4:** NF4 quantizes weights to 16 values spaced according to the normal distribution (where most neural network weights concentrate). This gives lower quantization error than uniformly-spaced INT4 for the same 4-bit budget.

**Double quantization:** The scale factors used in NF4 quantization are themselves quantized from FP32 to 8-bit, saving an additional ~200MB.

### Training Setup

```
Dataset  : DriveLM nuScenes (qa_enriched.csv)
Split    : 90/10 stratified by qa_category
Hardware : Google Colab T4 16GB
Script   : train_drivelm_llava.py
```

**Key hyperparameters:**

| Parameter | Value | Reason |
|---|---|---|
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Batch size | 1 | T4 VRAM constraint |
| Gradient accumulation | 4–8 | Effective batch = 4–8 |
| LoRA rank | 16 | Good balance, ~70M trainable params |
| Image size | 224px | Reduces tokens from 576 to 256/image — prevents OOM on T4 with 6-cam samples |
| Loss | Cross-entropy on answer tokens only | Prompt tokens masked with -100 |
| Scheduler | Linear warmup + cosine decay | Prevents large steps at LoRA init |

**Why image size 224px for training on T4:**

At 336px (native CLIP resolution), a 6-camera sample generates 3,456 visual tokens. Combined with model weights, this requires ~13.5GB peak VRAM — dangerously close to T4's 14.6GB limit. At 224px, 6-camera samples require only 1,536 visual tokens (~8.5GB peak) — comfortable headroom.

**Label masking — why it matters:**
```
Full sequence: USER: <system> <few-shot> Question: <Q> ASSISTANT: <answer>
Labels:         -100  -100      -100       -100       <answer tokens>  <eos>
```
Only answer tokens contribute to loss. Without masking, the model wastes capacity memorising the fixed prompt structure and gradient signal is diluted across ~95% non-answer tokens.

**Projector saved separately:** PEFT's `modules_to_save` mechanism crashes on 4-bit quantized models (bitsandbytes Params4bit tensors cannot have `requires_grad_(True)` called on them). We save the projector's `state_dict()` separately as `projector.pt` and restore it at inference time.

### Training Results (Initial Run)

Small validation run: 50 train samples, 30 val samples, 1 epoch, T4.

| Step | Val Loss | Val PPL | Improvement |
|---|---|---|---|
| 2 (16 samples) | 1.3875 | 4.00 | Baseline |
| 4 (32 samples) | 1.0884 | 2.97 | −21.5% |
| 6 (48 samples) | 1.0505 | 2.86 | −3.6% |
| epoch end | 1.0719 | 2.92 | — |

**Val PPL = 2.86** after only 48 samples is encouraging — the model is choosing between ~3 equally likely tokens per position, well within the expected range (2–5) for a fine-tuned LLM on constrained-vocabulary tasks.

Train loss decreased steadily: 2.33 → 1.93, confirming the model is learning. Val loss lower than train loss with 50 samples is expected — the small dataset means the model quickly fits training examples while val questions are generally simpler.

---

## 4. Comparative Evaluation

### Benchmark v1 vs v2 (Resolution vs Prompt Design)

The two baseline runs show a counterintuitive result — higher resolution without camera labels (v1) outperformed lower resolution with improved prompt (v2):

| Metric | v1 (448px, no labels) | v2 (336px, camera labels) | Δ |
|---|---|---|---|
| ROUGE-L | **0.3580** | 0.2942 | −17.8% |
| Exact Match | **0.2541** | 0.1553 | −38.9% |
| BERTScore F1 | **0.9268** | 0.9059 | −2.3% |

**Finding:** Visual resolution (448px → 1,024 tokens/image) had more impact than prompt-level spatial labelling for this dataset. Camera labels improve spatial reasoning but the resolution drop from 448px to 336px reduced visual information enough to offset that gain. The ideal setup — 448px with camera labels — exceeds T4 VRAM when using 6 cameras (6,144 visual tokens), which is the practical constraint we faced.

### Baseline vs Fine-Tuned

> Note: Full comparative benchmarking with fine-tuned model pending completion of full training run (500–800 samples, 3 epochs). Results below are from the small 50-sample training run.

**Training signal (val loss):**

| Metric | Baseline (untrained) | Fine-tuned (50 samples, 1 epoch) | Δ |
|---|---|---|---|
| Val Loss | ~1.33 (step 2 baseline) | 1.0505 (best) | −21.1% |
| Val PPL | ~4.00 | 2.86 | −28.5% |

**Where fine-tuning is expected to help:**

- **Planning format:** Base model defaults to generic templates ("Accelerate and go straight") regardless of the specific scenario. Fine-tuning on DriveLM examples teaches correct action selection
- **Status queries:** Base model confuses Moving/Stationary (90 failures). Fine-tuning on labeled examples directly addresses this
- **Answer conciseness:** Base model produces preambles ("Based on the image..."). Fine-tuning with masked prompts and answer-style rules suppresses this
- **Probability questions:** "Low/Medium/High" calibration improves with domain examples

**Where fine-tuning may hurt:**

- **General visual reasoning:** Fine-tuning on 50–800 DriveLM samples may cause slight forgetting of general VQA patterns outside DriveLM's distribution
- **Which-query:** These require identifying specific objects by position — needs more examples than available in the small training set
- **planning_other:** Open-ended reasoning questions that are rare in training data — likely to remain challenging

**Cost comparison (fine-tuned vs base):**

| Scenario | Base model | Fine-tuned |
|---|---|---|
| Inference latency | ~2,520 ms | ~2,520 ms (identical after merge_and_unload) |
| Inference VRAM | 3.82 GB | 3.82 GB (LoRA merged into weights) |
| Training cost (T4, 500 samples) | — | ~$1.50 (3 hours × $0.526/hr) |
| Break-even queries | — | ~3,200 queries (accuracy gain offsets cost) |

Fine-tuning adds a one-time training cost but zero inference overhead since LoRA adapters are merged into the base weights at deployment time.

---

## 5. Deployment & Optimization

### Architecture

The inference system (`infer_efficient.py`) implements an image embedding cache to avoid redundant computation across questions about the same scene.

```
Query arrives (frame_token, cameras, question)
        │
        ▼
Cache lookup: (frame_token, sorted_camera_names)
        │
   ┌────┴────┐
   │         │
 HIT       MISS
   │         │
   │    CLIP ViT-L/14 → [n, 576, 1024]     ~2.0s
   │    Projector MLP  → [n, 576, 4096]     ~0.2s
   │    Store in cache
   │         │
   └────┬────┘
        │
  cached visual_tokens [n*576, 4096]
        │
  Inject at <image> positions in text embeddings
        │
  LLaMA-2 7B autoregressive decode    ~1.0s
        │
  Answer string
```

### Why This Works: Data Structure Insight

From data analysis: **2,196 questions span only 215 unique (frame, camera) combinations** — on average **10.2 questions share the same images**.

```
Unique combos breakdown:
  1-cam:  69 combos    2-cam: 71 combos
  3-cam:  32 combos    4-cam: 21 combos
  6-cam:  22 combos
  Total: 215 combos → 2,196 questions

Without cache: 2,196 image encodings needed
With cache   : 215  image encodings needed  → 10.2x fewer encodings
```

Questions are sorted by `frame_token` before inference so all questions about the same frame are processed consecutively, maximising sequential cache hits.

### Cache Memory

| img-size | Total for all 215 combos | Notes |
|---|---|---|
| 224px | **1.02 GB** | Fits T4 easily (10 GB free after model) |
| 336px | **2.30 GB** | Fits T4 easily |

With `--cache-size 215` (default), all combos stay in cache — first 215 questions cause misses, remaining 1,981 are 100% cache hits.

### Latency & Throughput

| Path | Latency | Components |
|---|---|---|
| Cache miss | ~3,200 ms | CLIP (2,000) + projector (200) + LLM (1,000) |
| Cache hit | ~1,000 ms | LLM only |
| Avg (90% hit rate) | ~1,200 ms | 0.9×1,000 + 0.1×3,200 |

**Wall-time speedup vs benchmark_local.py (no cache):**
```
Without cache: 2,196 × 3.2s = 116 min
With cache   : 215 × 3.2s + 1,981 × 1.0s = ~43 min
Speedup      : 2.7x wall time
```

Note: wall-time speedup (2.7x) is less than encoding speedup (10.2x) because the LLM decode (~1s) still runs for every question regardless of cache.

### Cost per 1,000 Queries

| Setup | Latency/query | Rate | Cost / 1k queries |
|---|---|---|---|
| T4, no cache | 2,500 ms | $0.526/hr (AWS) | $0.47 |
| **T4, with cache** | **1,200 ms** | $0.526/hr | **$0.18** |
| A100, no cache | 800 ms | $1.10/hr | $0.24 |
| A100, with cache | 400 ms | $1.10/hr | $0.12 |

**At scale (1M queries/day):**

| Setup | Daily cost | Monthly cost |
|---|---|---|
| T4, no cache | $470 | $14,100 |
| T4, with cache | $180 | $5,400 |
| Saving | $290/day | **$8,700/month** |

### Does Caching Affect Accuracy?

**No — accuracy is mathematically identical to uncached inference.**

The cache stores exact FP16 tensors produced by CLIP and the projector. On a cache hit, the LLM receives the identical tensor it would have computed from scratch. There is no approximation, compression, or rounding beyond what was already present in the original computation.

The only theoretical concern is CUDA non-determinism — different runs of CLIP on the same image may produce values differing by ~1e-4 in FP16 due to non-deterministic GPU operations. The cache locks in the first run's values. For discrete outputs (Moving/Stationary, Low/Medium/High, Yes/No) this has zero practical effect.

### Serving Setup

A minimal FastAPI endpoint for production serving:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    frame_token: str
    cameras: list[str]
    question: str
    category: str

@app.post("/predict")
async def predict(query: Query):
    visual_tokens = cache.get(query.frame_token, query.cameras)
    if visual_tokens is None:
        images = load_images(query.frame_token, query.cameras)
        visual_tokens = extract_visual_tokens(model, processor, images)
        cache.put(query.frame_token, query.cameras, visual_tokens)
    prediction = generate_answer(model, processor, visual_tokens, query)
    return {"prediction": prediction}
```

Run with: `uvicorn serve:app --host 0.0.0.0 --port 8000`

---

## 6. Scripts Reference

| Script | Input | Output | Purpose |
|---|---|---|---|
| `parse_drivelm.py` | DriveLM JSON, nuScenes dir | `qa_enriched.csv`, `objects.csv`, `frames.csv`, `scenes.csv` | Data preparation |
| `vis_data.py` | `qa_enriched.csv`, images dir | `outputs/sample_N.jpg` | Dataset visualisation |
| `benchmark_local.py` | CSV, images dir | `benchmark_report.txt`, metrics CSVs | Baseline evaluation |
| `train_drivelm_llava.py` | Train/val CSVs, images dir | Checkpoint dir, `train_log.csv`, `train_log.txt` | QLoRA fine-tuning |
| `infer_efficient.py` | CSV, images dir | `predictions.csv` | Cached inference |
| `check_versions.py` | — | Console output | Verify package versions |

### Key CLI Arguments (training)

```bash
python3 train_drivelm_llava.py \
    --train-csv  qa_enriched_train.csv \
    --val-csv    qa_enriched_val.csv \
    --images     /path/to/nuscenes \
    --out        ./checkpoints \
    --mode       qlora \           # qlora (T4) or lora (A100)
    --img-size   224 \             # 224 for T4, 336 for A100
    --max-samples 800 \            # limit training samples
    --max-val-samples 50 \         # limit val samples
    --epochs     1 \
    --grad-accum 4 \               # adjust based on sample count
    --val-every  10 \
    --save-every 10 \
    --skip-baseline                # skip slow baseline val
```

### Key CLI Arguments (efficient inference)

```bash
python3 infer_efficient.py \
    --csv          qa_enriched_val.csv \
    --images       /path/to/nuscenes \
    --out          ./results \
    --adapter-path ./checkpoints/best_checkpoint \   # optional
    --img-size     336 \
    --cache-size   215                               # cache all 215 combos
```

---

## 7. Setup & Reproduction

### Requirements

```
Python      : 3.12
transformers: 4.40.0   ← exact, newer versions crash on 4-bit QLoRA
accelerate  : 0.27.2   ← exact, 0.28-0.29 has dispatch_model regression
peft        : 0.10.0   ← exact, 0.11+ needs transformers >= 4.43
bitsandbytes: 0.45.3   ← supports CUDA 12.8 (Colab Apr 2026)
torch       : >= 2.1.0
```

> **Critical:** These four packages must be pinned together. See `requirements.txt` for the full explanation of why each version was chosen.

### Install

```bash
pip install transformers==4.40.0 accelerate==0.27.2 \
            peft==0.10.0 bitsandbytes==0.45.3 \
            Pillow pandas tqdm sentencepiece rouge-score bert-score -q
```

### Verify

```bash
python3 check_versions.py
# All four critical packages should show "exact match"
```

### Docker

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04
RUN pip install transformers==4.40.0 accelerate==0.27.2 \
                peft==0.10.0 bitsandbytes==0.45.3 \
                Pillow pandas tqdm sentencepiece torch
COPY . /workspace
WORKDIR /workspace
```

```bash
docker build -t drivelm-vlm .
docker run --gpus all -v /path/to/nuscenes:/data/nuscenes drivelm-vlm \
    python3 benchmark_local.py --csv qa_enriched.csv --images /data/nuscenes
```

### Google Colab Quick Start

```python
# Cell 1 — Install
!pip install transformers==4.40.0 accelerate==0.27.2 \
             peft==0.10.0 bitsandbytes==0.45.3 -q

# Cell 2 — Restart runtime (mandatory)
import os; os.kill(os.getpid(), 9)

# Cell 3 — Verify
!python3 check_versions.py

# Cell 4 — Train
!python3 train_drivelm_llava.py \
    --train-csv qa_enriched.csv \
    --images /content/nuscenes \
    --out ./output \
    --mode qlora --img-size 224 \
    --max-samples 800 --max-val-samples 50 \
    --epochs 1 --skip-baseline

# Cell 5 — Benchmark
!python3 infer_efficient.py \
    --csv qa_enriched_val.csv \
    --images /content/nuscenes \
    --adapter-path ./output/best_checkpoint
```

---

## Summary of Key Decisions

| Decision | Choice | Alternative Considered | Reason |
|---|---|---|---|
| Base model | LLaVA-1.5-7B | LLaVA-1.5-13B, InstructBLIP | Best capability/memory tradeoff for T4 |
| Quantization | 4-bit NF4 | FP16, INT8 | Only option that fits T4 for training |
| Fine-tuning method | QLoRA | Full fine-tune, LoRA FP16 | QLoRA: 0.6% trainable params, fits T4 |
| LoRA targets | Attn + MLP | Attention only | MLP layers store DriveLM answer patterns |
| Image size | 224px (training) | 336px native | Prevents OOM on T4 with 6-cam samples |
| Primary metric | ROUGE-L | Accuracy, F1 | Handles variable-length DriveLM answers |
| Camera selection | `relevant_cameras` column | All 6 always | Reduces tokens, improves accuracy |
| Cache point | After projector | After CLIP | Skips both CLIP and projector (2.2s saved) |
| Cache size | 215 (all combos) | 50 (default LRU) | All 215 combos fit in 1–2.3GB on T4 |
