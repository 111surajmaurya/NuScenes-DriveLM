"""
DriveLM Local VLM Benchmark  —  LLaVA-1.5-7B
=============================================

KEY DESIGN DECISIONS (read before editing):

  CAMERA SELECTION STRATEGY
  ─────────────────────────
  relevant_cameras == relevant_image_paths cameras in every row (0 mismatches
  across 2196 rows verified).  They carry identical camera lists.

  However relevant_image_paths stores full file paths keyed by camera name,
  while relevant_cameras is a clean semicolon-separated list of camera names.

  Strategy used here:
    • Parse WHICH cameras to use  →  from relevant_cameras  (clean, explicit)
    • Resolve actual file paths   →  from all_image_paths   (always has all 6)
  This decouples "which cameras are semantically relevant" from "where the file is".

  WHY relevant_cameras is the right source:
    • perception  status/visual_description questions: 1 camera  (e.g. "front camera")
    • perception  object_enumeration / yes_no:         6 cameras (scene-wide)
    • prediction:                                      1-3 cameras (specific objects)
    • planning    object-specific questions:           1-2 cameras
    • planning    scene-level questions:               6 cameras
    • behavior:                                        always 6 cameras (need all
                                                       for speed + direction cues)

  IMAGE → MODEL BINDING
  ─────────────────────
  Each <image> token is immediately preceded by its camera label inline:
      [Front Camera]: <image>
      [Front Left Camera]: <image>
  This gives LLaVA unambiguous binding between visual tokens and camera position.
  Without this, the model sees anonymous image blobs with no spatial context.

Install:
  pip install transformers accelerate pillow pandas tqdm rouge-score bert-score sentencepiece protobuf bitsandbytes

Usage:
  python3 benchmark_local.py \\
      --csv     qa_enriched.csv \\
      --images  ./data/nuscenes \\
      --out     ./benchmark_results \\
      [--limit  100]

  python3 benchmark_local.py --csv qa_enriched.csv --images ./data/nuscenes --dry-run
  python3 benchmark_local.py --csv qa_enriched.csv --images ./data/nuscenes --detect
"""

import os, re, sys, csv, time, argparse, random, gc
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from rouge_score import rouge_scorer


# ════════════════════════════════════════════════════════════════════════════
# MODEL CONFIG
# ════════════════════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    'name'       : 'llava-1.5-7b-hf',
    'hf_id'      : 'llava-hf/llava-1.5-7b-hf',
    'type'       : 'llava',
    'quantize'   : '4bit',
    'description': 'LLaVA-1.5 7B (4-bit nf4)',
}


# ════════════════════════════════════════════════════════════════════════════
# GPU DETECTION
# ════════════════════════════════════════════════════════════════════════════

def detect_gpu() -> dict:
    if not torch.cuda.is_available():
        return {'available': False, 'vram_gb': 0, 'name': 'CPU'}
    props = torch.cuda.get_device_properties(0)
    return {
        'available': True,
        'vram_gb'  : props.total_memory / 1024**3,
        'name'     : props.name,
        'device'   : 'cuda',
    }


def print_gpu_info():
    gpu = detect_gpu()
    print(f'\n{"="*60}\n  GPU Detection\n{"="*60}')
    if gpu['available']:
        print(f'  GPU   : {gpu["name"]}')
        print(f'  VRAM  : {gpu["vram_gb"]:.1f} GB')
        print(f'  Model : {MODEL_CONFIG["hf_id"]}')
    else:
        print('  No GPU found — CPU inference will be very slow.')
    print(f'{"="*60}')


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

ALL_CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',
]

# Human-readable label shown inline in prompt next to each <image> token
CAMERA_LABEL = {
    'CAM_FRONT'      : 'Front Camera',
    'CAM_FRONT_LEFT' : 'Front Left Camera',
    'CAM_FRONT_RIGHT': 'Front Right Camera',
    'CAM_BACK'       : 'Back Camera',
    'CAM_BACK_LEFT'  : 'Back Left Camera',
    'CAM_BACK_RIGHT' : 'Back Right Camera',
}

# Fallback strategy if relevant_cameras column is missing/empty
# (should not happen with this dataset, but kept as safety net)
FALLBACK_STRATEGY = {
    'perception': ['CAM_FRONT'],
    'prediction': ['CAM_FRONT'],
    'planning'  : ['CAM_FRONT'],
    'behavior'  : ALL_CAMERAS,
}

SYSTEM_PROMPT = (
    "You are an autonomous driving assistant analyzing surround-view camera images. "
    "Answer concisely and accurately based only on what is visible in the images. "
    "Do not hallucinate objects or events not visible."
)


# ════════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES  —  one block per QA category
# ════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = {

    'perception': """\
Examples of correct perception answers:
Q: What are objects to the back of the ego car?
A: There are many pedestrians and two cars behind the ego car.

Q: What is the status of the pedestrians that are to the back of the ego car?
A: Many pedestrians are standing, and many are moving.

Q: What is the observed status of the vehicle in back camera?
A: Moving.

""",

    'prediction': """\
Examples of correct prediction answers:
Q: In this scenario, what object is most likely to consider the traffic element in front camera?
A: None.

Q: Are the object in front camera and the object in back camera traffic signs?
A: Neither is a traffic sign.

Q: What object would consider the vehicle in front camera to be most relevant to its decision?
A: The ego vehicle.

""",

    'planning': """\
Examples of correct planning answers:
Q: What actions taken by the ego vehicle can lead to a collision with the vehicle in front camera?
A: Accelerate and go straight actions taken by the ego vehicle can lead to a collision with the vehicle in front camera.

Q: What actions could the ego vehicle take based on the vehicle in back camera? Why take this action and what's the probability?
A: The action is to keep going at the same speed. The reason is to maintain a safe distance, which is high.

""",

    'behavior': """\
Examples of correct behavior answers:
Q: Predict the behavior of the ego vehicle.
A: The ego vehicle is going straight. The ego vehicle is driving fast.

Q: Predict the behavior of the ego vehicle.
A: The ego vehicle is slightly steering to the right. The ego vehicle is driving slowly.

""",
}

ANSWER_STYLE_RULES = """\
Answer style rules — follow these strictly:
- Be concise. Match the length and style of the examples above.
- For object list questions: use "There are X [objects] to the [direction]" format.
- For status questions: one word if possible — "Stationary", "Moving", "Parked".
- For yes/no questions: answer only "Yes" or "No" then one short sentence if needed.
- For probability questions: answer only "Low", "Medium", or "High".
- For behavior: exactly 1-2 short sentences — "The ego vehicle is [action]."
- For planning: state the action and reason concisely in 1-2 sentences.
- If the answer is nothing/none: say "None."
- Do NOT explain your reasoning unless the question asks why.
- Do NOT describe the image beyond what is needed to answer.
- Do NOT start your answer with "Based on the image" or similar preambles.
"""


# ════════════════════════════════════════════════════════════════════════════
# IMAGE PATH PARSING
# ════════════════════════════════════════════════════════════════════════════

def parse_camera_paths(value: str) -> dict:
    """
    Parse 'CAM_FRONT:path/to/img.jpg | CAM_BACK:path/to/img.jpg ...'
    into {camera_name: path_string}.
    """
    paths = {}
    value = str(value)
    if value and value.lower() != 'nan':
        for part in value.split(' | '):
            if ':' in part:
                cam, path = part.split(':', 1)
                paths[cam.strip()] = path.strip()
    return paths


def parse_relevant_cameras(value: str) -> list:
    """
    Parse 'CAM_FRONT; CAM_BACK_LEFT' into ['CAM_FRONT', 'CAM_BACK_LEFT'].
    Returns empty list if value is nan/empty.
    """
    value = str(value)
    if not value or value.lower() == 'nan':
        return []
    return [c.strip() for c in value.split(';') if c.strip()]


# ════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING
# ════════════════════════════════════════════════════════════════════════════

def resolve_image_path(image_path: str, nusc_root: str) -> Path | None:
    path = Path(image_path)
    if path.exists():
        return path
    cleaned = str(path).replace('../nuscenes/', '')
    alt = Path(nusc_root) / cleaned
    return alt if alt.exists() else None


def load_pil_image(image_path: str, nusc_root: str,
                   max_size: int = 448) -> Image.Image | None:
    resolved = resolve_image_path(image_path, nusc_root)
    if not resolved:
        return None
    img = Image.open(resolved).convert('RGB')
    w, h = img.size
    scale = max_size / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def get_images_for_row(row: pd.Series, category: str,
                       nusc_root: str, max_size: int = 448) -> dict:
    """
    Return OrderedDict {camera_name: PIL.Image} for cameras needed by this QA.

    STRATEGY:
      1. Use relevant_cameras column to decide WHICH cameras are semantically
         relevant to this specific question. This column is precise:
           - "front camera" questions  → only CAM_FRONT
           - scene-wide questions      → all 6 cameras
           - behavior                  → always all 6
      2. Resolve file paths from all_image_paths (always has all 6 paths).
         This decouples camera selection from path storage.
      3. Fallback: if relevant_cameras is empty, use FALLBACK_STRATEGY.

    Why not use relevant_image_paths for path resolution?
      relevant_image_paths only stores paths for the relevant cameras, so if
      relevant_cameras is correct then relevant_image_paths has the same cameras
      anyway. Using all_image_paths as path source is more robust (always has
      all 6) and makes camera selection logic independent of path storage.
    """
    # Step 1: determine which cameras to use
    relevant_cams = parse_relevant_cameras(row.get('relevant_cameras', ''))
    if not relevant_cams:
        relevant_cams = FALLBACK_STRATEGY.get(category, ['CAM_FRONT'])
        print(f'  [WARN] Empty relevant_cameras for {category} row, '
              f'using fallback: {relevant_cams}')

    # Step 2: get path lookup from all_image_paths (always has all 6)
    all_paths = parse_camera_paths(row.get('all_image_paths', ''))

    # Step 3: load images in canonical camera order (preserves spatial layout)
    result = {}
    missing_cams = []
    for cam in ALL_CAMERAS:                # canonical order, not arbitrary dict order
        if cam not in relevant_cams:
            continue
        path = all_paths.get(cam)
        if not path:
            missing_cams.append(cam)
            continue
        img = load_pil_image(path, nusc_root, max_size)
        if img:
            result[cam] = img
        else:
            missing_cams.append(cam)

    if missing_cams:
        print(f'  [WARN] Missing images for cameras: {missing_cams} '
              f'(category={category}, frame={str(row.get("frame_token",""))[:8]})')

    return result


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_model():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from transformers import BitsAndBytesConfig

    hf_id = MODEL_CONFIG['hf_id']
    print(f'\n  Loading {hf_id} ...')

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )

    processor = AutoProcessor.from_pretrained(hf_id, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        hf_id,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor, MODEL_CONFIG


# ════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_full_prompt(question: str, cam_names: list, category: str) -> str:
    """
    Build the full prompt string with <image> tokens interleaved inline with
    camera labels so LLaVA knows exactly which image token = which camera.

    Structure:
      USER:
        <system prompt>
        <few-shot examples>
        <answer style rules>

        Surround-view camera images:
        [Front Camera]: <image>
        [Front Left Camera]: <image>
        ...

        Question: <question>

      ASSISTANT:

    WHY inline labels matter:
      LLaVA maps each <image> token positionally to the PIL images list passed
      to the processor. Image 0 → first <image> token, image 1 → second, etc.
      By writing "[Front Left Camera]: <image>" we bind the label to the token
      at the same position, giving the model unambiguous spatial context.
      Without this, images are anonymous blobs and the model cannot reliably
      reference "front left camera" in its answer.
    """
    few_shot = FEW_SHOT_EXAMPLES.get(category, '')

    # Build image section with inline labels — one label:token pair per camera
    image_lines = []
    for cam in cam_names:
        label = CAMERA_LABEL.get(cam, cam.replace('_', ' ').title())
        image_lines.append(f'[{label}]: <image>')
    image_section = '\n'.join(image_lines)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{few_shot}"
        f"{ANSWER_STYLE_RULES}\n"
        f"Surround-view camera images provided:\n"
        f"{image_section}\n\n"
        f"Question: {question}\n\n"
    )

    return f"USER: {prompt}ASSISTANT:"


# ════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ════════════════════════════════════════════════════════════════════════════

def split_sentences(text: str) -> list:
    """
    Safe sentence splitter that does NOT break on decimal numbers like "2.5".
    Splits on '. ' or '.' at end of string only when followed by uppercase
    or end of string, guarding against decimal splits.
    """
    # Split on period followed by space+uppercase or end-of-string
    parts = re.split(r'(?<!\d)\.(?!\d)(?:\s+(?=[A-Z])|\s*$)', text)
    return [p.strip() for p in parts if p.strip()]


def postprocess_answer(pred: str, category: str) -> str:
    """
    Clean up model output to match DriveLM answer style.
    - Strips preamble phrases
    - Extracts categorical values (Low/Medium/High, Yes/No)
    - Truncates to 2 sentences for perception/behavior/planning
    """
    if not pred:
        return pred

    pred = pred.strip()

    # Strip common preamble patterns
    preambles = [
        r'^based on the (image|images|camera|cameras)[,.]?\s*',
        r'^looking at the (image|images|camera|cameras)[,.]?\s*',
        r'^from the (image|images|camera|cameras)[,.]?\s*',
        r'^in the (image|images|camera|cameras)[,.]?\s*',
        r'^the (image|images) (show|shows|reveal|reveals)[s]?[,.]?\s*',
        r'^i can (see|observe)[,.]?\s*',
        r'^it appears (that\s*)?',
        r'^it seems (that\s*)?',
    ]
    for pattern in preambles:
        cleaned = re.sub(pattern, '', pred, flags=re.IGNORECASE).strip()
        if cleaned and cleaned != pred:
            pred = cleaned[0].upper() + cleaned[1:]
            break

    # Probability questions — extract Low / Medium / High
    prob_keywords = ['probability', 'collid', 'likelihood', 'chance']
    if any(kw in pred.lower() for kw in prob_keywords):
        for word in ['Low', 'Medium', 'High']:
            if word.lower() in pred.lower():
                return word + '.'

    # Yes/No/None — keep only first sentence
    if re.match(r'^(yes|no|none)[,. ]', pred, re.IGNORECASE):
        sentences = split_sentences(pred)
        return (sentences[0] if sentences else pred) + '.'

    # Category-based truncation — 2 sentences max
    if category in ('behavior', 'perception', 'planning'):
        sentences = split_sentences(pred)
        if len(sentences) > 2:
            return '. '.join(sentences[:2]) + '.'

    # General fallback — truncate if > 50 words
    if len(pred.split()) > 50:
        sentences = split_sentences(pred)
        return '. '.join(sentences[:2]) + '.'

    return pred


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def infer_llava(model, processor, images: dict,
                question: str, category: str,
                max_new_tokens: int = 150) -> str:
    """
    Run LLaVA inference.

    images: OrderedDict {camera_name: PIL.Image} — ORDER MATTERS.
      The PIL images list must be in the same order as the <image> tokens
      in the prompt. Both are built from the same cam_names list so they
      are always aligned.

    IMAGE COUNT GUARD: asserts that <image> token count == len(pil_images).
      A mismatch causes silent visual-token misalignment or a cryptic crash.
    """
    cam_names  = list(images.keys())   # ordered camera names
    pil_images = list(images.values()) # same order as cam_names

    full_prompt = build_full_prompt(question, cam_names, category)

    # Safety check: token count must exactly match image count
    token_count = full_prompt.count('<image>')
    assert token_count == len(pil_images), (
        f'<image> token count ({token_count}) != image count ({len(pil_images)}). '
        f'Cameras: {cam_names}'
    )

    inputs = processor(
        text=full_prompt,
        images=pil_images,   # always pass as list (processor handles single too)
        return_tensors='pt',
        padding=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def run_inference(model, processor, cfg: dict,
                  images: dict, question: str,
                  category: str) -> tuple[str, float]:
    t0   = time.perf_counter()
    pred = infer_llava(model, processor, images, question, category)
    pred = postprocess_answer(pred, category)
    return pred, (time.perf_counter() - t0) * 1000


# ════════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def rouge_l(pred: str, ref: str, scorer) -> float:
    return scorer.score(ref, pred)['rougeL'].fmeasure


def compute_bert_scores(preds: list, refs: list) -> list:
    from bert_score import score as bscore
    if not preds:
        return []
    _, _, F1 = bscore(
        preds, refs, lang='en', verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    return F1.tolist()


def classify_failure(pred: str, ref: str, category: str, rl: float) -> str | None:
    """
    Classify failure mode for low-scoring predictions (ROUGE-L < 0.3).

    CAMERA WORD FIX: original code used single words {'front','back','left','right'}
    which matched non-camera phrases like "steering to the left" or "turn right".
    Now uses camera-specific phrases to avoid false C_wrong_camera classification.
    """
    if rl >= 0.3:
        return None
    pred_n = normalize(pred)
    ref_n  = normalize(ref)

    status_words = {'stationary', 'moving', 'parked', 'stopped', 'driving', 'standing'}

    # Use multi-word camera phrases instead of single direction words
    camera_phrases = {
        'front camera', 'back camera', 'front left camera',
        'front right camera', 'back left camera', 'back right camera',
        'front-left camera', 'front-right camera',
    }

    pred_status  = status_words & set(pred_n.split())
    ref_status   = status_words & set(ref_n.split())

    def has_cam_phrase(text):
        return any(phrase in text for phrase in camera_phrases)

    if len(pred_n.split()) > 15 and len(set(pred_n.split()) & set(ref_n.split())) < 3:
        return 'A_hallucination'
    if pred_status and ref_status and pred_status != ref_status:
        return 'B_wrong_status'
    if has_cam_phrase(pred_n) and has_cam_phrase(ref_n) and pred_n != ref_n:
        # only flag if the camera mentioned differs, not just any direction word
        pred_cams = {p for p in camera_phrases if p in pred_n}
        ref_cams  = {p for p in camera_phrases if p in ref_n}
        if pred_cams and ref_cams and pred_cams != ref_cams:
            return 'C_wrong_camera'
    if len(pred_n.split()) < len(ref_n.split()) * 0.4:
        return 'D_incomplete'
    if category in ('planning', 'behavior'):
        return 'E_planning_error'
    return 'F_other'


# ════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_benchmark(df: pd.DataFrame, model, processor, cfg: dict,
                  nusc_root: str, output_dir: str,
                  limit: int | None = None,
                  categories: list | None = None,
                  max_image_size: int = 448) -> tuple[pd.DataFrame, int]:

    os.makedirs(output_dir, exist_ok=True)

    if categories:
        df = df[df['qa_category'].isin(categories)].copy()

    if limit and limit < len(df):
        df = (df.groupby('qa_category', group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), max(1, limit // df['qa_category'].nunique())),
                    random_state=42))
                .reset_index(drop=True))
        print(f'  Sampled {len(df)} QA pairs (limit={limit})')

    scorer  = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = []
    skipped = 0

    print(f'\n  Benchmarking {len(df)} QA pairs with {cfg["hf_id"]} ...\n')

    for _, row in tqdm(df.iterrows(), total=len(df)):
        category = str(row['qa_category'])
        question = str(row['question_readable'])
        answer   = str(row['answer_readable'])

        images = get_images_for_row(row, category, nusc_root, max_image_size)

        if not images:
            skipped += 1
            results.append({
                'scene_token'  : row.get('scene_token', ''),
                'frame_token'  : row.get('frame_token', ''),
                'qa_category'  : category,
                'question_type': row.get('question_type', ''),
                'question'     : question,
                'reference'    : answer,
                'prediction'   : '',
                'rouge_l'      : None,
                'exact_match'  : None,
                'bert_score'   : None,
                'latency_ms'   : None,
                'num_images'   : 0,
                'cameras_used' : '',
                'failure_mode' : 'NO_IMAGE',
                'vram_used_gb' : None,
            })
            continue

        try:
            pred, latency_ms = run_inference(
                model, processor, cfg, images, question, category
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            vram_used = torch.cuda.memory_allocated() / 1024**3 \
                        if torch.cuda.is_available() else 0
            skipped += 1
            print(f'\n  OOM on frame {str(row.get("frame_token",""))[:8]} — skipping')
            results.append({
                'scene_token'  : row.get('scene_token', ''),
                'frame_token'  : row.get('frame_token', ''),
                'qa_category'  : category,
                'question_type': row.get('question_type', ''),
                'question'     : question,
                'reference'    : answer,
                'prediction'   : '',
                'rouge_l'      : None,
                'exact_match'  : None,
                'bert_score'   : None,
                'latency_ms'   : None,
                'num_images'   : len(images),
                'cameras_used' : '; '.join(images.keys()),
                'failure_mode' : 'OOM',
                'vram_used_gb' : round(vram_used, 2),
            })
            for img in images.values():
                img.close()
            continue

        vram_used = torch.cuda.memory_allocated() / 1024**3 \
                    if torch.cuda.is_available() else 0

        rl = rouge_l(pred, answer, scorer)
        em = exact_match(pred, answer)
        fm = classify_failure(pred, answer, category, rl)

        results.append({
            'scene_token'  : row.get('scene_token', ''),
            'frame_token'  : row.get('frame_token', ''),
            'qa_category'  : category,
            'question_type': row.get('question_type', ''),
            'question'     : question,
            'reference'    : answer,
            'prediction'   : pred,
            'rouge_l'      : round(rl, 4),
            'exact_match'  : em,
            'bert_score'   : None,          # filled in batch below
            'latency_ms'   : round(latency_ms, 1),
            'num_images'   : len(images),
            'cameras_used' : '; '.join(images.keys()),
            'failure_mode' : fm,
            'vram_used_gb' : round(vram_used, 2),
        })
        for img in images.values():
            img.close()

    # Batch BERTScore
    print('\n  Computing BERTScore ...')
    valid_idx = [i for i, r in enumerate(results) if r['prediction']]
    if valid_idx:
        preds  = [results[i]['prediction'] for i in valid_idx]
        refs   = [results[i]['reference']  for i in valid_idx]
        scores = compute_bert_scores(preds, refs)
        for i, idx in enumerate(valid_idx):
            results[idx]['bert_score'] = round(scores[i], 4)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(output_dir, 'raw_results.csv'),
        index=False, quoting=csv.QUOTE_ALL,
    )
    return results_df, skipped


# ════════════════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════════════════

def generate_report(df: pd.DataFrame, model_key: str,
                    cfg: dict, skipped: int, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    valid = df[df['rouge_l'].notna()].copy()
    lines = []

    def h(t): lines.append(f'\n{"="*65}\n  {t}\n{"="*65}')
    def s(t): lines.append(f'\n  -- {t}')

    h(f'DriveLM Local VLM Benchmark  --  {model_key}')
    lines += [
        f'  Model      : {cfg["hf_id"]}',
        f'  Quantize   : {cfg.get("quantize", "none")}',
        f'  Date       : {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'  Total QAs  : {len(df)}',
        f'  Evaluated  : {len(valid)}',
        f'  Skipped    : {skipped}  (image missing or OOM)',
    ]
    print('\n'.join(lines[-6:]))

    s('OVERALL METRICS')
    peak_vram = valid['vram_used_gb'].max() if 'vram_used_gb' in valid.columns else 0
    metrics = {
        'ROUGE-L (primary)'  : valid['rouge_l'].mean(),
        'Exact Match'        : valid['exact_match'].mean(),
        'BERTScore F1'       : valid['bert_score'].mean(),
        'Avg Latency ms'     : valid['latency_ms'].mean(),
        'P95 Latency ms'     : valid['latency_ms'].quantile(0.95),
        'Peak VRAM GB'       : peak_vram,
    }
    for k, v in metrics.items():
        line = f'    {k:<22}: {v:.4f}'
        lines.append(line)
        print(line)

    s('METRICS BY QA CATEGORY')
    cat = (valid.groupby('qa_category').agg(
        n           = ('rouge_l', 'count'),
        rouge_l     = ('rouge_l', 'mean'),
        exact_match = ('exact_match', 'mean'),
        bert_score  = ('bert_score', 'mean'),
        latency_ms  = ('latency_ms', 'mean'),
        avg_images  = ('num_images', 'mean'),
    ).round(4).reset_index())
    lines.append(cat.to_string(index=False))
    print(cat.to_string(index=False))
    cat.to_csv(os.path.join(output_dir, 'metrics_by_category.csv'),
               index=False, quoting=csv.QUOTE_ALL)

    s('METRICS BY QUESTION TYPE  (hardest first)')
    qt = (valid.groupby('question_type').agg(
        n          = ('rouge_l', 'count'),
        rouge_l    = ('rouge_l', 'mean'),
        bert_score = ('bert_score', 'mean'),
    ).round(4).sort_values('rouge_l').reset_index())
    lines.append(qt.to_string(index=False))
    print(qt.to_string(index=False))
    qt.to_csv(os.path.join(output_dir, 'metrics_by_qtype.csv'),
              index=False, quoting=csv.QUOTE_ALL)

    s('LATENCY DISTRIBUTION')
    lat = valid['latency_ms'].describe(percentiles=[.5, .75, .9, .95, .99]).round(1)
    lines.append(lat.to_string())
    print(lat.to_string())
    lat_by_cat = (valid.groupby('qa_category')['latency_ms']
                       .agg(['mean', 'median', 'max']).round(1).reset_index())
    lines.append('\n  By category:\n' + lat_by_cat.to_string(index=False))
    lat_by_cat.to_csv(os.path.join(output_dir, 'latency_by_category.csv'),
                      index=False, quoting=csv.QUOTE_ALL)

    s('FAILURE MODE DISTRIBUTION')
    failures = valid[valid['failure_mode'].notna()]
    fm = failures['failure_mode'].value_counts().reset_index()
    fm.columns = ['failure_mode', 'count']
    fm['pct_of_evaluated'] = (fm['count'] / len(valid) * 100).round(1)
    fm['description'] = fm['failure_mode'].map({
        'A_hallucination' : 'Model hallucinates objects not in image',
        'B_wrong_status'  : 'Stationary/Moving confused',
        'C_wrong_camera'  : 'Wrong camera viewpoint referenced',
        'D_incomplete'    : 'Answer too short / misses objects',
        'E_planning_error': 'Wrong ego behavior or direction',
        'F_other'         : 'Other mismatch',
        'NO_IMAGE'        : 'Image file not found',
        'OOM'             : 'Out of GPU memory',
    })
    lines.append(fm.to_string(index=False))
    print(fm.to_string(index=False))
    fm.to_csv(os.path.join(output_dir, 'failure_modes.csv'),
              index=False, quoting=csv.QUOTE_ALL)

    s('REPRESENTATIVE SUCCESSES  (ROUGE-L >= 0.7, 2 per category)')
    succ = (valid[valid['rouge_l'] >= 0.7]
            .sort_values('rouge_l', ascending=False)
            .groupby('qa_category').head(2))
    succ_rows = []
    for _, r in succ.iterrows():
        lines += [
            f'\n  [{r["qa_category"].upper()}]  RL={r["rouge_l"]:.3f}  BS={r["bert_score"]:.3f}',
            f'  Q  : {r["question"][:110]}',
            f'  REF: {r["reference"][:110]}',
            f'  PRD: {r["prediction"][:110]}',
        ]
        succ_rows.append({
            'category'  : r['qa_category'], 'rouge_l': r['rouge_l'],
            'bert_score': r['bert_score'],  'question': r['question'],
            'reference' : r['reference'],   'prediction': r['prediction'],
        })
    pd.DataFrame(succ_rows).to_csv(os.path.join(output_dir, 'successes.csv'),
                                   index=False, quoting=csv.QUOTE_ALL)

    s('REPRESENTATIVE FAILURES  (2 per failure type)')
    fail_rows = []
    for ftype in ['A_hallucination', 'B_wrong_status', 'C_wrong_camera',
                  'D_incomplete', 'E_planning_error']:
        for _, r in valid[valid['failure_mode'] == ftype].head(2).iterrows():
            lines += [
                f'\n  [{ftype}]  [{r["qa_category"].upper()}]  RL={r["rouge_l"]:.3f}',
                f'  Q  : {r["question"][:110]}',
                f'  REF: {r["reference"][:110]}',
                f'  PRD: {r["prediction"][:110]}',
            ]
            fail_rows.append({
                'failure_mode': ftype, 'category': r['qa_category'],
                'rouge_l': r['rouge_l'], 'question': r['question'],
                'reference': r['reference'], 'prediction': r['prediction'],
            })
    pd.DataFrame(fail_rows).to_csv(os.path.join(output_dir, 'failures.csv'),
                                   index=False, quoting=csv.QUOTE_ALL)

    report_path = os.path.join(output_dir, 'benchmark_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\n  Report   -> {report_path}')
    print(f'  All CSVs -> {output_dir}/')


# ════════════════════════════════════════════════════════════════════════════
# DRY RUN
# ════════════════════════════════════════════════════════════════════════════

def dry_run(df: pd.DataFrame, nusc_root: str, output_dir: str):
    print('\n  DRY RUN -- no model loaded\n')
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    found = missing = 0

    for _, row in df.iterrows():
        cat = str(row['qa_category'])
        imgs = get_images_for_row(row, cat, nusc_root, max_size=32)
        relevant_cams = parse_relevant_cameras(row.get('relevant_cameras', ''))
        if not relevant_cams:
            relevant_cams = FALLBACK_STRATEGY.get(cat, ['CAM_FRONT'])

        n_expected = len(relevant_cams)
        n_found    = len(imgs)
        n_missing  = n_expected - n_found
        found   += n_found
        missing += max(0, n_missing)

        for img in imgs.values():
            img.close()

        rows.append({
            'qa_category'      : cat,
            'n_images_expected': n_expected,
            'n_images_found'   : n_found,
            'n_images_missing' : max(0, n_missing),
            'relevant_cameras' : '; '.join(relevant_cams),
        })

    dr = pd.DataFrame(rows)
    print(dr.groupby('qa_category').agg(
        total_qa            = ('n_images_found', 'count'),
        avg_expected        = ('n_images_expected', 'mean'),
        avg_imgs_found      = ('n_images_found', 'mean'),
        avg_imgs_missing    = ('n_images_missing', 'mean'),
    ).round(2).to_string())
    print(f'\n  Total images found  : {found}')
    print(f'  Total images missing: {missing}')
    dr.to_csv(os.path.join(output_dir, 'dry_run_summary.csv'),
              index=False, quoting=csv.QUOTE_ALL)
    print(f'\n  Saved -> {output_dir}/dry_run_summary.csv')


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',        required=True)
    p.add_argument('--images',     required=True)
    p.add_argument('--out',        default='./benchmark_results')
    p.add_argument('--limit',      type=int, default=None)
    p.add_argument('--categories', nargs='+',
                   default=['perception', 'prediction', 'planning', 'behavior'])
    p.add_argument('--img-size',   type=int, default=448)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--detect',     action='store_true')
    p.add_argument('--dry-run',    action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)

    if args.detect:
        print_gpu_info()
        sys.exit(0)

    gpu = detect_gpu()
    print(f'\n{"="*65}')
    print(f'  DriveLM Local VLM Benchmark')
    print(f'{"="*65}')
    print(f'  Model    : {MODEL_CONFIG["hf_id"]}  ({MODEL_CONFIG["description"]})')
    print(f'  GPU      : {gpu["name"]}  ({gpu["vram_gb"]:.1f} GB VRAM)')
    print(f'  CSV      : {args.csv}')
    print(f'  Images   : {args.images}')
    print(f'  Img size : {args.img_size}px')
    print(f'  Limit    : {args.limit or "all"}')
    print(f'{"="*65}')

    df = pd.read_csv(args.csv)
    print(f'\n  Loaded {len(df)} QA rows')

    if args.dry_run:
        dry_run(df, args.images, args.out)
        sys.exit(0)

    model, processor, cfg = load_model()

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f'  VRAM used after load: {alloc:.2f} GB')

    results_df, skipped = run_benchmark(
        df             = df,
        model          = model,
        processor      = processor,
        cfg            = cfg,
        nusc_root      = args.images,
        output_dir     = args.out,
        limit          = args.limit,
        categories     = args.categories,
        max_image_size = args.img_size,
    )

    generate_report(
        df         = results_df,
        model_key  = MODEL_CONFIG['name'],
        cfg        = cfg,
        skipped    = skipped,
        output_dir = args.out,
    )

    print(f'\n  Done. Results in: {args.out}/')
