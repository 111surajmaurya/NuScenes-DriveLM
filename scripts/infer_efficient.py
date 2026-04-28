import os, re, gc, csv, time, argparse
from pathlib import Path
from collections import OrderedDict
from typing import Optional

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

MODEL_HF_ID = 'llava-hf/llava-1.5-7b-hf'

ALL_CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',
]

CAMERA_LABEL = {
    'CAM_FRONT'      : 'Front Camera',
    'CAM_FRONT_LEFT' : 'Front Left Camera',
    'CAM_FRONT_RIGHT': 'Front Right Camera',
    'CAM_BACK'       : 'Back Camera',
    'CAM_BACK_LEFT'  : 'Back Left Camera',
    'CAM_BACK_RIGHT' : 'Back Right Camera',
}

FALLBACK_CAMERAS = {
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

FEW_SHOT_EXAMPLES = {
    'perception': """\
Examples of correct perception answers:
Q: What are objects to the back of the ego car?
A: There are many pedestrians and two cars behind the ego car.

Q: What is the observed status of the vehicle in back camera?
A: Moving.

""",
    'prediction': """\
Examples of correct prediction answers:
Q: What is the future state of the vehicle in front camera?
A: Stationary.

Q: Are the object in front camera and the object in back camera traffic signs?
A: Neither is a traffic sign.

""",
    'planning': """\
Examples of correct planning answers:
Q: What actions taken by the ego vehicle can lead to a collision with the vehicle in front camera?
A: Accelerate and go straight.

Q: What is the probability of colliding with the vehicle in front camera?
A: Low.

""",
    'behavior': """\
Examples of correct behavior answers:
Q: Predict the behavior of the ego vehicle.
A: The ego vehicle is going straight. The ego vehicle is driving fast.

""",
}

ANSWER_STYLE_RULES = """\
Answer style rules:
- Be concise. Match the length and style of the examples above.
- For status: one word — "Stationary", "Moving", "Parked".
- For yes/no: "Yes" or "No" then one short sentence if needed.
- For probability: "Low", "Medium", or "High".
- For behavior: 1-2 sentences — "The ego vehicle is [action]."
- If nothing/none: "None."
- Do NOT start with "Based on the image".
"""


# ════════════════════════════════════════════════════════════════════════════
# IMAGE EMBEDDING CACHE  (LRU)
# ════════════════════════════════════════════════════════════════════════════

class VisualTokenCache:
    """
    LRU cache mapping (frame_token, camera_set) → visual_tokens tensor.

    Stores visual tokens AFTER the projector so both CLIP and projector
    are skipped on cache hits. Each tensor is [n_cams * 576, 4096].

    Memory per entry:
      1-cam : 1 × 576 × 4096 × 2 bytes (FP16) =  4.7 MB
      3-cam : 3 × 576 × 4096 × 2 bytes         = 14.2 MB
      6-cam : 6 × 576 × 4096 × 2 bytes         = 28.3 MB
    max_entries=50 → worst case ~1.4 GB (all 6-cam)
    """

    def __init__(self, max_entries: int = 50):
        self.cache       = OrderedDict()
        self.max_entries = max_entries
        self.hits        = 0
        self.misses      = 0

    def _key(self, frame_token: str, cam_names: list) -> tuple:
        return (str(frame_token), tuple(sorted(cam_names)))

    def get(self, frame_token: str, cam_names: list):
        k = self._key(frame_token, cam_names)
        if k in self.cache:
            self.cache.move_to_end(k)
            self.hits += 1
            return self.cache[k]
        self.misses += 1
        return None

    def put(self, frame_token: str, cam_names: list, visual_tokens):
        k = self._key(frame_token, cam_names)
        if k in self.cache:
            self.cache.move_to_end(k)
        else:
            if len(self.cache) >= self.max_entries:
                _, evicted = self.cache.popitem(last=False)
                del evicted
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self.cache[k] = visual_tokens

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            'hits'       : self.hits,
            'misses'     : self.misses,
            'hit_rate'   : self.hits / max(1, total),
            'entries'    : len(self.cache),
            'speedup_est': f'{total / max(1, self.misses):.1f}x',
        }


# ════════════════════════════════════════════════════════════════════════════
# CSV / IMAGE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def parse_camera_paths(value: str) -> dict:
    paths = {}
    for part in str(value).split(' | '):
        if ':' in part:
            cam, path = part.split(':', 1)
            paths[cam.strip()] = path.strip()
    return paths


def parse_relevant_cameras(value: str) -> list:
    v = str(value)
    if not v or v.lower() == 'nan':
        return []
    return [c.strip() for c in v.split(';') if c.strip()]


def load_pil_images(row: pd.Series, category: str,
                    nusc_root: str, max_size: int) -> dict:
    # planning and behavior always use all 6 cameras — same fix as benchmark script
    if category in ('planning', 'behavior'):
        relevant = list(ALL_CAMERAS)
    else:
        relevant = parse_relevant_cameras(row.get('relevant_cameras', ''))
        if not relevant:
            relevant = FALLBACK_CAMERAS.get(category, ['CAM_FRONT'])
    all_paths = parse_camera_paths(row.get('all_image_paths', ''))

    result = {}
    for cam in ALL_CAMERAS:
        if cam not in relevant:
            continue
        path = all_paths.get(cam)
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            p = Path(nusc_root) / str(path).replace('../nuscenes/', '')
        if not p.exists():
            continue
        img = Image.open(p).convert('RGB')
        w, h = img.size
        scale = max_size / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        result[cam] = img
    return result


# ════════════════════════════════════════════════════════════════════════════
# PROMPT
# ════════════════════════════════════════════════════════════════════════════

def build_prompt(question: str, cam_names: list, category: str) -> str:
    few_shot    = FEW_SHOT_EXAMPLES.get(category, '')
    image_lines = [
        f'[{CAMERA_LABEL.get(c, c.replace("_"," ").title())}]: <image>'
        for c in cam_names
    ]
    image_section = chr(10).join(image_lines)
    user_content = (
        f"{image_section}\n\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"{few_shot}"
        f"{ANSWER_STYLE_RULES}\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return f"USER: {user_content}\nASSISTANT:"


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_model(adapter_path: str = None):
    from transformers import LlavaForConditionalGeneration
    from transformers import LlavaProcessor, CLIPImageProcessor, AutoTokenizer
    from transformers import BitsAndBytesConfig

    print(f'\n  Loading {MODEL_HF_ID} ...')

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )

    image_processor = CLIPImageProcessor.from_pretrained(MODEL_HF_ID)
    tokenizer       = AutoTokenizer.from_pretrained(MODEL_HF_ID, use_fast=False)
    processor       = LlavaProcessor(
        image_processor=image_processor, tokenizer=tokenizer
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_HF_ID,
        quantization_config = quant_cfg,
        torch_dtype         = torch.float16,
        low_cpu_mem_usage   = True,
    )

    if adapter_path:
        from peft import PeftModel
        print(f'  Loading adapter: {adapter_path}')
        model = PeftModel.from_pretrained(
            model, adapter_path,
            is_trainable=False
        )
        print('  LoRA adapters loaded (kept separate — preserves 4-bit memory)')

        proj_path = os.path.join(adapter_path, 'projector.pt')
        if os.path.exists(proj_path):
            proj_sd = torch.load(proj_path, map_location='cpu', weights_only=True)
            for attr in ['base_model.model.multi_modal_projector',
                         'multi_modal_projector',
                         'model.multi_modal_projector']:
                try:
                    proj = model
                    for part in attr.split('.'): proj = getattr(proj, part)
                    proj.load_state_dict(
                        {k.replace('original_module.', ''): v
                         for k, v in proj_sd.items()}, strict=False
                    )
                    print(f'  Projector loaded')
                    break
                except AttributeError:
                    continue

    model.eval()

    if torch.cuda.is_available():
        print(f'  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

    return model, processor


# ════════════════════════════════════════════════════════════════════════════
# VISUAL TOKEN EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_visual_tokens(model, processor, pil_images: list,
                          device) -> torch.Tensor:
    """
    CLIP vision tower + projector only. Returns [n_cams * 576, 4096].
    This is the expensive step cached by VisualTokenCache.

    Pipeline:
      pixel_values [n,3,H,W]
        → CLIP ViT hidden states [-2] [n,577,1024]
        → remove CLS token         [n,576,1024]
        → projector (linear+GELU+linear) [n,576,4096]
        → flatten                  [n*576,4096]
    """
    pixel_values = processor.image_processor(
        images=pil_images, return_tensors='pt'
    ).pixel_values.to(device, dtype=torch.float16)

    # Find vision tower and projector by walking named_modules.
    # Resolve the correct base depending on whether LoRA is still wrapped.
    if hasattr(model, 'base_model'):
        base = model.base_model.model
    elif hasattr(model, 'language_model'):
        base = model   # LlavaForConditionalGeneration — walk the whole thing
    else:
        base = model
    vision_tower = None
    mm_projector  = None
    for _, mod in base.named_modules():
        cls = type(mod).__name__
        if vision_tower is None and (
            'CLIPVisionModel' in cls or 'SiglipVisionModel' in cls
        ):
            vision_tower = mod
        if mm_projector is None and (
            'LlavaMultiModalProjector' in cls or 'MultiModalProjector' in cls
        ):
            mm_projector = mod

    # CLIP forward
    vision_out   = vision_tower(pixel_values, output_hidden_states=True)
    # LLaVA-1.5 uses second-to-last hidden state, then removes CLS token
    img_features = vision_out.hidden_states[-2][:, 1:, :]  # [n,576,1024]

    # Projector forward
    vis_tokens = mm_projector(img_features)   # [n,576,4096]
    n, seq, dim = vis_tokens.shape
    return vis_tokens.reshape(n * seq, dim).half()  # [n*576,4096]


# ════════════════════════════════════════════════════════════════════════════
# CACHED INFERENCE
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_one(model, processor, row: pd.Series, category: str,
              question: str, cam_names: list, pil_images: list,
              cache: VisualTokenCache, device,
              max_new_tokens: int = 100) -> tuple:
    """
    Infer one question, reusing visual tokens from cache when available.

    Cache hit path  (~LLM time only, no image encoding):
      get cached visual_tokens
      build text embeddings
      inject visual_tokens at <image> positions
      model.generate(inputs_embeds=combined)

    Cache miss path (~CLIP + projector + LLM):
      encode images → visual_tokens
      store in cache
      same injection + generate
    """
    t0 = time.perf_counter()
    frame = str(row.get('frame_token', ''))

    # Try cache
    visual_tokens = cache.get(frame, cam_names)
    cache_hit     = visual_tokens is not None

    if not cache_hit:
        visual_tokens = extract_visual_tokens(
            model, processor, pil_images, device
        )
        cache.put(frame, cam_names, visual_tokens)

    # Tokenize text prompt
    prompt = build_prompt(question, cam_names, category)
    tok    = processor.tokenizer(
        prompt, return_tensors='pt',
        truncation=True, max_length=512,
    ).to(device)
    input_ids = tok['input_ids']

    if hasattr(model, 'base_model'):
        # PeftModel still wrapping (is_trainable=False path)
        llava = model.base_model.model
    else:
        # Merged model or plain base model
        llava = model

    # Get the LLaMA language model — this has .generate() and .get_input_embeddings()
    if hasattr(llava, 'language_model'):
        lm = llava.language_model          # LlamaForCausalLM
    elif hasattr(llava, 'model'):
        lm = llava.model                   # fallback
    else:
        lm = llava

    embed_fn = lm.get_input_embeddings()
    text_emb = embed_fn(input_ids)  # [1, text_len, 4096]

    # Build combined embeddings:
    # Replace each <image> token (id=32000) with 576 visual tokens
    IMAGE_TOKEN_ID  = 32000
    n_image_tokens  = sum(1 for pos in range(input_ids.shape[1])
                          if input_ids[0, pos].item() == IMAGE_TOKEN_ID)
    TOKENS_PER_CAM  = (visual_tokens.shape[0] // n_image_tokens
                       if n_image_tokens > 0 else 576)
    new_emb         = []
    vis_cam_idx     = 0

    for pos in range(input_ids.shape[1]):
        if input_ids[0, pos].item() == IMAGE_TOKEN_ID:
            start = vis_cam_idx * TOKENS_PER_CAM
            end   = start + TOKENS_PER_CAM
            if end <= visual_tokens.shape[0]:
                cam_tok = visual_tokens[start:end]           # [576, 4096]
                new_emb.append(cam_tok.unsqueeze(0))         # [1, 576, 4096]
                vis_cam_idx += 1
            else:
                new_emb.append(text_emb[:, pos:pos+1, :])
        else:
            new_emb.append(text_emb[:, pos:pos+1, :])

    combined_emb = torch.cat(new_emb, dim=1)  # [1, full_seq, 4096]
    attn_mask    = torch.ones(
        combined_emb.shape[:2], dtype=torch.long, device=device
    )

    try:
        out_ids = lm.generate(
            inputs_embeds  = combined_emb,
            attention_mask = attn_mask,
            max_new_tokens = max_new_tokens,
            do_sample      = False,
            temperature    = None,
            pad_token_id   = processor.tokenizer.eos_token_id,
        )
        pred = processor.tokenizer.decode(
            out_ids[0], skip_special_tokens=True
        ).strip()
    except Exception as e:
        pred = ''
        print(f'  [ERROR] {e}')

    # Strip preamble
    for pat in [r'^based on the \w+[,.]?\s*',
                r'^looking at the \w+[,.]?\s*',
                r'^i can (see|observe)[,.]?\s*']:
        cleaned = re.sub(pat, '', pred, flags=re.IGNORECASE).strip()
        if cleaned and cleaned != pred:
            pred = cleaned[0].upper() + cleaned[1:]
            break

    ms = (time.perf_counter() - t0) * 1000
    return pred, ms, cache_hit


# ════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_inference(df, model, processor, nusc_root, output_dir,
                  img_size=336, max_new_tokens=100,
                  cache_size=50, dry_run=False):

    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device if model else 'cpu'
    cache  = VisualTokenCache(max_entries=cache_size)

    # Sort by frame+cameras so cache hits are consecutive
    df = df.copy()
    df['_sort_key'] = (df['frame_token'].astype(str) + '|' +
                       df['relevant_cameras'].astype(str))
    df = df.sort_values('_sort_key').drop(columns='_sort_key').reset_index(drop=True)

    unique_combos = df.groupby(['frame_token','relevant_cameras']).ngroups
    print(f'\n  Questions        : {len(df)}')
    print(f'  Unique img combos: {unique_combos}')
    print(f'  Expected speedup : ~{len(df)/unique_combos:.1f}x\n')

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cat      = str(row['qa_category'])
        question = str(row['question_readable'])
        answer   = str(row['answer_readable'])
        frame    = str(row.get('frame_token', ''))

        images    = load_pil_images(row, cat, nusc_root, img_size)
        cam_names = list(images.keys())
        pil_list  = list(images.values())

        if not images:
            results.append({
                'frame_token': frame, 'category': cat,
                'question': question, 'reference': answer,
                'prediction': '', 'latency_ms': None,
                'cache_hit': False, 'error': 'NO_IMAGE',
            })
            continue

        if dry_run:
            results.append({
                'frame_token': frame, 'category': cat,
                'question': question, 'reference': answer,
                'prediction': '[DRY RUN]', 'latency_ms': 0,
                'cache_hit': False, 'error': '',
            })
            for img in pil_list: img.close()
            continue

        try:
            pred, ms, hit = infer_one(
                model, processor, row, cat, question,
                cam_names, pil_list, cache, device, max_new_tokens
            )
        except Exception as e:
            pred, ms, hit = '', 0, False
            print(f'  [ERROR] {e}')

        results.append({
            'frame_token': frame, 'category': cat,
            'question': question, 'reference': answer,
            'prediction': pred, 'latency_ms': round(ms, 1),
            'cache_hit': hit, 'error': '',
        })
        for img in pil_list: img.close()

    # Save
    out_df   = pd.DataFrame(results)
    out_path = os.path.join(output_dir, 'predictions.csv')
    out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)

    # Stats
    stats = cache.stats()
    valid = out_df[out_df['prediction'].notna() & (out_df['prediction'] != '')]

    print(f'\n{"═"*55}')
    print(f'  Done')
    print(f'{"═"*55}')
    print(f'  Total            : {len(df)}')
    print(f'  Cache hit rate   : {stats["hit_rate"]*100:.1f}%  '
          f'({stats["hits"]} hits / {stats["misses"]} misses)')
    print(f'  Effective speedup: {stats["speedup_est"]}')
    if len(valid):
        avg  = valid['latency_ms'].mean()
        hits = valid[valid['cache_hit']]['latency_ms']
        miss = valid[~valid['cache_hit']]['latency_ms']
        print(f'  Avg latency      : {avg:.0f} ms/question')
        if len(hits): print(f'  Cache hit  latency: {hits.mean():.0f} ms')
        if len(miss): print(f'  Cache miss latency: {miss.mean():.0f} ms')
    print(f'  Results → {out_path}')
    print(f'{"═"*55}')
    return out_df


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',          required=True)
    p.add_argument('--images',       required=True)
    p.add_argument('--out',          default='./inference_results')
    p.add_argument('--adapter-path', default=None)
    p.add_argument('--limit',        type=int, default=None)
    p.add_argument('--categories',   nargs='+',
                   default=['perception','prediction','planning','behavior'])
    p.add_argument('--img-size',     type=int, default=448)
    p.add_argument('--max-tokens',   type=int, default=100)
    p.add_argument('--dry-run',      action='store_true')
    p.add_argument('--cache-size',   type=int, default=215,
                   help='Max cached (frame, camera_set) entries. '
                        '215 = cache ALL unique combos in qa_enriched '
                        '(1.0 GB at 224px, 2.3 GB at 336px). '
                        'Reduce only if VRAM is very tight.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f'\n{"═"*65}')
    print(f'  DriveLM Efficient Inference (with image embedding cache)')
    print(f'{"═"*65}')
    print(f'  CSV          : {args.csv}')
    print(f'  Images       : {args.images}')
    print(f'  Adapter      : {args.adapter_path or "none (base model)"}')
    print(f'  Img size     : {args.img_size}px')
    print(f'  Cache size   : {args.cache_size} entries')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU          : {props.name} ({props.total_memory/1024**3:.1f} GB)')
    print(f'{"═"*65}')

    df = pd.read_csv(args.csv)
    if args.categories:
        df = df[df['qa_category'].isin(args.categories)]
    if args.limit:
        df = df.head(args.limit)
    print(f'\n  Loaded {len(df)} questions')

    if not args.dry_run:
        model, processor = load_model(adapter_path=args.adapter_path)
    else:
        model = processor = None
        print('  [DRY RUN] Skipping model load')

    run_inference(
        df            = df,
        model         = model,
        processor     = processor,
        nusc_root     = args.images,
        output_dir    = args.out,
        img_size      = args.img_size,
        max_new_tokens= args.max_tokens,
        cache_size    = args.cache_size,
        dry_run       = args.dry_run,
    )
