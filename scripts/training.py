import os, re, gc, csv, math, time, argparse, random
from pathlib import Path
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

# LoRA applied to both attention AND MLP layers of the LLM
LORA_TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',    # attention
    'gate_proj', 'up_proj', 'down_proj',         # MLP / FFN
]


# ════════════════════════════════════════════════════════════════════════════
# PROMPTS  (identical to benchmark_local.py for train/eval consistency)
# ════════════════════════════════════════════════════════════════════════════

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
A: Accelerate and go straight actions taken by the ego vehicle can lead to a collision with the vehicle in front camera.

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
- For status questions: one word if possible — "Stationary", "Moving", "Parked".
- For yes/no questions: "Yes" or "No" then one short sentence if needed.
- For probability: "Low", "Medium", or "High".
- For behavior: 1-2 sentences — "The ego vehicle is [action]."
- If nothing/none: say "None."
- Do NOT start with "Based on the image" or similar preambles.
"""


# ════════════════════════════════════════════════════════════════════════════
# CSV / IMAGE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def parse_camera_paths(value: str) -> dict:
    paths = {}
    value = str(value)
    if value and value.lower() != 'nan':
        for part in value.split(' | '):
            if ':' in part:
                cam, path = part.split(':', 1)
                paths[cam.strip()] = path.strip()
    return paths


def parse_relevant_cameras(value: str) -> list:
    value = str(value)
    if not value or value.lower() == 'nan':
        return []
    return [c.strip() for c in value.split(';') if c.strip()]


def resolve_image_path(image_path: str, nusc_root: str) -> Optional[Path]:
    path = Path(image_path)
    if path.exists():
        return path
    cleaned = str(path).replace('../nuscenes/', '')
    alt = Path(nusc_root) / cleaned
    return alt if alt.exists() else None


def load_pil_image(image_path: str, nusc_root: str,
                   max_size: int = 336) -> Optional[Image.Image]:
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
                       nusc_root: str, max_size: int = 336) -> dict:
    """
    Returns {camera_name: PIL.Image} in canonical spatial order.
    Uses relevant_cameras for WHICH cameras, all_image_paths for file paths.
    (relevant_cameras == relevant_image_paths cameras in all rows — verified.)
    """
    relevant_cams = parse_relevant_cameras(row.get('relevant_cameras', ''))
    if not relevant_cams:
        relevant_cams = FALLBACK_CAMERAS.get(category, ['CAM_FRONT'])

    all_paths = parse_camera_paths(row.get('all_image_paths', ''))

    result = {}
    for cam in ALL_CAMERAS:
        if cam not in relevant_cams:
            continue
        path = all_paths.get(cam)
        if not path:
            continue
        img = load_pil_image(path, nusc_root, max_size)
        if img:
            result[cam] = img
    return result


# ════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_prompt(question: str, cam_names: list, category: str,
                 answer: str = '') -> str:
    """
    Build the full conversation string.

    Training (answer provided):
      USER: <system> <few-shot> <rules>
            [Front Camera]: <image>
            [Front Left Camera]: <image>
            Question: <question>
      ASSISTANT: <answer>

    Inference (no answer):
      USER: ... ASSISTANT:

    Inline camera labels bind each <image> token to its spatial position
    so the model knows which visual token = which camera direction.
    """
    few_shot = FEW_SHOT_EXAMPLES.get(category, '')

    image_lines = []
    for cam in cam_names:
        label = CAMERA_LABEL.get(cam, cam.replace('_', ' ').title())
        image_lines.append(f'[{label}]: <image>')
    image_section = '\n'.join(image_lines)

    user_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{few_shot}"
        f"{ANSWER_STYLE_RULES}\n"
        f"Surround-view camera images:\n"
        f"{image_section}\n\n"
        f"Question: {question}"
    )

    if answer:
        return f"USER: {user_content}\nASSISTANT: {answer}"
    else:
        return f"USER: {user_content}\nASSISTANT:"


# ════════════════════════════════════════════════════════════════════════════
# TRAIN / VAL SPLIT
# ════════════════════════════════════════════════════════════════════════════

def split_dataframe(df: pd.DataFrame, val_ratio: float = 0.1,
                    seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 90/10 split by qa_category.

    Why stratified (not random):
      behavior only has 22 rows. A pure random split might assign all 22 to
      train, leaving val completely blind to behavior performance.
      Stratified split guarantees every category appears in both splits.

    Example result with val_ratio=0.1:
      perception : 970 → 873 train +  97 val
      prediction : 702 → 632 train +  70 val
      planning   : 502 → 452 train +  50 val
      behavior   :  22 →  20 train +   2 val
    """
    train_parts, val_parts = [], []
    for cat in df['qa_category'].unique():
        sub   = df[df['qa_category'] == cat].sample(frac=1, random_state=seed)
        n_val = max(1, int(len(sub) * val_ratio))
        val_parts.append(sub.iloc[:n_val])
        train_parts.append(sub.iloc[n_val:])

    train_df = (pd.concat(train_parts)
                  .sample(frac=1, random_state=seed)
                  .reset_index(drop=True))
    val_df   = (pd.concat(val_parts)
                  .sample(frac=1, random_state=seed)
                  .reset_index(drop=True))
    return train_df, val_df


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class DriveLMDataset(torch.utils.data.Dataset):
    """
    Loads DriveLM QA rows and serves (prompt_text, images, metadata).

    __getitem__ returns dict:
      prompt_text : full USER:.../ASSISTANT: <answer> string
      images      : list[PIL.Image] in cam order (matches <image> tokens)
      category    : qa_category string
      question    : question string
      answer      : ground truth answer string
    """

    def __init__(self, df: pd.DataFrame, nusc_root: str,
                 max_size: int = 336, split: str = 'train'):
        df = df[df['answer_readable'].notna()].copy()
        df = df[df['question_readable'].notna()].copy()
        self.df        = df.reset_index(drop=True)
        self.nusc_root = nusc_root
        self.max_size  = max_size
        self.split     = split

        print(f'\n  [{split}] {len(self.df)} samples')
        for cat, n in self.df['qa_category'].value_counts().items():
            print(f'    {cat:<12}: {n}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        category = str(row['qa_category'])
        question = str(row['question_readable'])
        answer   = str(row['answer_readable'])

        images    = get_images_for_row(row, category, self.nusc_root, self.max_size)
        cam_names = list(images.keys())
        pil_list  = list(images.values())

        prompt_text = build_prompt(question, cam_names, category, answer)

        return {
            'prompt_text': prompt_text,
            'images'     : pil_list,
            'cam_names'  : cam_names,
            'category'   : category,
            'question'   : question,
            'answer'     : answer,
        }


# ════════════════════════════════════════════════════════════════════════════
# COLLATOR
# ════════════════════════════════════════════════════════════════════════════

class DriveLMCollator:
    """
    Tokenizes a batch and builds answer-only loss labels.

    LABEL MASKING DETAIL:
      Full token sequence:
        USER: <system> <rules> [Front Camera]: <image> ... Q: <q> ASSISTANT: <answer> <eos>

      Labels (what model is trained to predict):
        -100  -100  ...  -100  -100  -100  ...  <answer tokens>  <eos>

      Steps:
        1. Clone input_ids → labels
        2. Find last "ASSISTANT:" token substring in each sequence
           (last because few-shot examples in prompt also contain "ASSISTANT:")
        3. Set labels[:assistant_end] = -100
        4. Set labels[padding positions] = -100
        5. PyTorch CrossEntropyLoss skips all -100 positions automatically

      Result: loss and gradients computed ONLY over answer tokens.
              The model learns to generate answers, not to predict prompts.

    MULTI-IMAGE BATCHING:
      Each sample may have 1–6 images. We flatten all images across the
      batch into one list. The processor maps them positionally to <image>
      tokens. Both lists are built from the same cam_names order, so
      position alignment is guaranteed.
    """

    def __init__(self, processor, max_length: int = 3584):
        self.processor  = processor
        self.max_length = max_length
        self.ignore_idx = -100

        # Cache ASSISTANT: token ids — used to find masking boundary
        self._assistant_ids = processor.tokenizer.encode(
            'ASSISTANT:', add_special_tokens=False
        )

    def __call__(self, batch):
        texts       = [item['prompt_text'] for item in batch]
        images      = [item['images']      for item in batch]
        flat_images = []
        for img_list in images:
            flat_images.extend(img_list)

        encoding = self.processor(
            text           = texts,
            images         = flat_images if flat_images else None,
            return_tensors = 'pt',
            padding        = True,
            truncation     = True,
            max_length     = self.max_length,
        )

        input_ids      = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        pixel_values   = encoding.get('pixel_values')

        labels = input_ids.clone()

        for i in range(labels.shape[0]):
            ids           = input_ids[i].tolist()
            assistant_pos = self._find_last_subseq(ids, self._assistant_ids)

            if assistant_pos != -1:
                mask_end = assistant_pos + len(self._assistant_ids)
                labels[i, :mask_end] = self.ignore_idx
            else:
                labels[i, :] = self.ignore_idx
                print(f'  [WARN] ASSISTANT: not found in sequence {i} — masking all')

        labels[attention_mask == 0] = self.ignore_idx

        result = {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'labels'        : labels,
        }
        if pixel_values is not None:
            result['pixel_values'] = pixel_values
        return result

    @staticmethod
    def _find_last_subseq(sequence: list, subseq: list) -> int:
        """Last occurrence of subseq in sequence. Returns -1 if not found."""
        n, m = len(sequence), len(subseq)
        for i in range(n - m, -1, -1):
            if sequence[i:i + m] == subseq:
                return i
        return -1


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_model_and_processor(mode: str = 'qlora', lora_r: int = 16,
                             cpu_test: bool = False):
    """
    Load LLaVA-1.5-7B with LoRA or QLoRA configuration.

    mode='qlora' : 4-bit NF4 base + BF16 LoRA adapters  (~12 GB VRAM, GPU only)
    mode='lora'  : FP16 base + FP16 LoRA adapters        (~22 GB VRAM, GPU only)
    cpu_test=True: FP32, no quantization, CPU-safe. Use to verify the full
                   pipeline (data loading, collator, forward pass, loss,
                   label masking, LoRA injection) without needing a GPU.
                   Loads the full model in FP32 on CPU (~28 GB RAM needed).
                   Use --max-samples 4 --epochs 1 to keep it quick.

    What gets frozen / trained:
      vision_tower          → frozen   (no grad, saves 307M param memory)
      multi_modal_projector → trained  (full, 20M params, high impact)
      language_model        → LoRA     (A+B matrices ~70M params at r=16)
    """
    from transformers import LlavaForConditionalGeneration, LlavaProcessor
    from transformers import CLIPImageProcessor, AutoTokenizer
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f'\n  Loading {MODEL_HF_ID}  (mode={mode}, lora_r={lora_r}) ...')

    # cpu_test mode: skip all quantization — bitsandbytes is GPU-only
    if cpu_test:
        print('  [CPU TEST MODE] Skipping quantization, loading FP32 on CPU')
        mode = 'lora'   # treat as lora so prepare_model_for_kbit_training is skipped

    quant_cfg = None
    if mode == 'qlora' and not cpu_test:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = 'nf4',
        )

    # Load processor components separately and construct LlavaProcessor directly.
    #
    # WHY NOT AutoProcessor.from_pretrained():
    #   The Hub config for llava-1.5-7b-hf now contains extra fields
    #   (image_token, patch_size, num_additional_image_tokens,
    #    vision_feature_select_strategy) that were added after the model
    #   was originally released. AutoProcessor.from_pretrained() downloads
    #   this config and tries to pass ALL fields to LlavaProcessor.__init__()
    #   via cls(*args, **processor_dict). Older transformers versions whose
    #   LlavaProcessor.__init__ does not accept these kwargs crash with:
    #     TypeError: LlavaProcessor.__init__() got unexpected keyword 'image_token'
    #
    #   Constructing manually bypasses the config dict entirely and works
    #   on any transformers version that has LlavaProcessor (>= 4.36).
    image_processor = CLIPImageProcessor.from_pretrained(MODEL_HF_ID)
    tokenizer       = AutoTokenizer.from_pretrained(MODEL_HF_ID, use_fast=False)
    processor       = LlavaProcessor(image_processor=image_processor,
                                     tokenizer=tokenizer)
    processor.tokenizer.pad_token    = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = 'right'

    # device_map / dtype handling
    # ─────────────────────────────────────────────────────────────────────
    # cpu_test : FP32, no device_map — stays on CPU, no GPU needed.
    #
    # qlora    : NO device_map. With transformers==4.40.0 (pinned in
    #            requirements.txt), from_pretrained does NOT auto-infer
    #            device_map for quantized models, so dispatch_model() is
    #            never called and bitsandbytes handles GPU placement itself.
    #            This is the only clean solution — transformers >= 4.38
    #            auto-infers device_map internally, always triggering
    #            dispatch_model() → model.to() → crash for 4-bit models.
    #
    # lora     : device_map='auto' — no quantization, works normally.
    if cpu_test:
        load_kwargs = dict(
            torch_dtype       = torch.float32,   # FP32, CPU safe
            low_cpu_mem_usage = True,
            # no device_map — model stays on CPU
        )
    elif mode == 'qlora':
        load_kwargs = dict(
            low_cpu_mem_usage   = True,
            quantization_config = quant_cfg,
            torch_dtype         = torch.bfloat16,
            # NO device_map — transformers 4.40.0 does not auto-infer it
            # for quantized models, so dispatch_model() is never triggered.
            # bitsandbytes places the model on GPU by itself.
        )
    else:
        load_kwargs = dict(
            device_map        = 'auto',
            low_cpu_mem_usage = True,
            torch_dtype       = torch.float16,
        )

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_HF_ID, **load_kwargs
    )

    # ── Inspect actual model structure ───────────────────────────────────
    # Instead of hardcoding model.model.X or model.X (which changes across
    # transformers versions), we walk model.named_modules() and find paths
    # by class name. This is version-agnostic and fails loudly if anything
    # is missing, giving you the actual class names to debug with.
    vision_tower_path = None
    mm_projector_path = None

    for mod_name, mod in model.named_modules():
        cls = type(mod).__name__
        if vision_tower_path is None and (
            'CLIPVisionModel'   in cls or
            'SiglipVisionModel' in cls or
            'VisionTower'       in cls
        ):
            vision_tower_path = mod_name
        if mm_projector_path is None and (
            'LlavaMultiModalProjector' in cls or
            'MultiModalProjector'      in cls
        ):
            mm_projector_path = mod_name

    print(f'  vision_tower path      : {vision_tower_path!r}')
    print(f'  mm_projector path      : {mm_projector_path!r}')

    if vision_tower_path is None or mm_projector_path is None:
        # Print all top-level module names to help debug
        print('  [ERROR] Could not auto-detect submodule paths.')
        print('  Top-level submodules found:')
        for n, m in model.named_children():
            print(f'    {n}: {type(m).__name__}')
        print('  All named modules (first 30):')
        for i, (n, m) in enumerate(model.named_modules()):
            if i >= 30: break
            print(f'    {n}: {type(m).__name__}')
        raise RuntimeError(
            'Could not find vision_tower or multi_modal_projector in model. '
            'Check the module list above and update the class name patterns '
            'in inspect block above.'
        )

    def _get_submodule(model, dotted_path):
        obj = model
        for part in dotted_path.split('.'):
            obj = getattr(obj, part)
        return obj

    vision_tower = _get_submodule(model, vision_tower_path)
    mm_projector = _get_submodule(model, mm_projector_path)

    # Build modules_to_save: only nn.Linear layers inside the projector.
    # The projector has: linear_1, act (GELU activation), linear_2.
    # modules_to_save must NOT include activation layers — they have no
    # trainable parameters and peft's ModulesToSaveWrapper calls
    # .requires_grad_(True) on all their children which crashes on
    # non-float buffers (e.g. integer indices in some activations).
    # Filter to only submodules that actually have trainable parameters.
    import torch.nn as nn
    projector_save = [
        f'{mm_projector_path}.{child_name}'
        for child_name, child_mod in mm_projector.named_children()
        if isinstance(child_mod, nn.Linear)
    ]
    print(f'  projector_save paths   : {projector_save}')

    # ── Freeze vision_tower ───────────────────────────────────────────────
    for param in vision_tower.parameters():
        param.requires_grad = False
    print('  vision_tower          : FROZEN')

    # ── Prepare for k-bit training (QLoRA) ───────────────────────────────
    # MUST happen BEFORE setting requires_grad on projector.
    # prepare_model_for_kbit_training sets requires_grad=False on everything
    # first, then casts LayerNorm to FP32. We re-enable the projector after.
    if mode == 'qlora':
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing    = True,
            gradient_checkpointing_kwargs = {'use_reentrant': False},
            # use_reentrant=False is stable with peft+bitsandbytes on Colab T4.
            # False is safer for complex custom autograd but LLaVA does not
            # need it. True = classic checkpointing, fully correct here.
        )

    # ── Full-train multi_modal_projector ──────────────────────────────────
    # Re-enable gradients on projector linear layers (float params only).
    for param in mm_projector.parameters():
        if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
            param.requires_grad = True
    print('  multi_modal_projector : FULLY TRAINED (20M params)')

    # ── Apply LoRA to language_model only ────────────────────────────────
    # NO modules_to_save: peft's ModulesToSaveWrapper calls
    # module.requires_grad_(True) on the whole module including bitsandbytes
    # Params4bit buffers which are not float — causes RuntimeError.
    # Projector is already trainable via manual requires_grad loop above.
    # Projector weights are saved separately via projector.pt in save_checkpoint.
    lora_cfg = LoraConfig(
        r              = lora_r,
        lora_alpha     = lora_r * 2,
        target_modules = LORA_TARGET_MODULES,
        lora_dropout   = 0.05,
        bias           = 'none',
        task_type      = 'CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print(f'  language_model        : LoRA on {LORA_TARGET_MODULES}')

    if torch.cuda.is_available():
        print(f'  VRAM after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

    # Return mm_projector_path so save_checkpoint can save projector separately
    return model, processor, mm_projector_path


# ════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_validation(model, val_loader, device) -> tuple[float, float]:
    """
    Compute val_loss and val_perplexity on the full validation set.

    Strategy:
      1. torch.no_grad() (decorator): prevents gradient computation.
         This is sufficient to stop autograd — we do NOT call model.eval()
         because that disables gradient checkpointing, causing all activations
         to be stored → OOM on T4.
      2. Disable gradient checkpointing during val: with no_grad active,
         checkpointing serves no purpose and triggers a warning:
         'None of inputs have requires_grad=True. Gradients will be None'
         We disable it before val and re-enable after.
      3. Clear CUDA cache before each batch to prevent fragmentation OOM.
    """
    # Disable gradient checkpointing during val — suppresses warning:
    # 'None of inputs have requires_grad=True. Gradients will be None'
    try:
        model.gradient_checkpointing_disable()
        _gc_was_enabled = True
    except Exception:
        _gc_was_enabled = False
    model.eval()

    total_loss    = 0.0
    total_batches = 0
    oom_count     = 0

    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        try:
            outputs = model(
                input_ids      = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                pixel_values   = batch.get('pixel_values'),
                labels         = batch['labels'],
            )
            if not torch.isnan(outputs.loss):
                total_loss    += outputs.loss.item()
                total_batches += 1
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    # Restore training mode and gradient checkpointing
    model.train()
    if _gc_was_enabled:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
        except Exception:
            pass

    if oom_count > 0:
        print(f'  [WARN] Val OOM on {oom_count} batches — '
              f'{total_batches} batches succeeded')

    if total_batches == 0:
        print('  [ERROR] All val batches OOMed — val_loss=inf')
        print('  Try: --img-size 224 to reduce memory usage.')
        return float('inf'), float('inf')

    avg_loss = total_loss / total_batches
    ppl      = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, processor, save_dir: str,
                    mm_projector_path: str = None,
                    metadata: dict = None):
    """
    Save adapter weights + processor. Base model NOT saved (loaded from HF).

    Saved files (~300 MB total):
      adapter_model.safetensors  LoRA A,B + projector weights
      adapter_config.json        LoRA hyperparameters
      tokenizer + processor files

    To load for inference later:
      from peft import PeftModel
      from transformers import LlavaForConditionalGeneration, AutoProcessor
      base      = LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf')
      model     = PeftModel.from_pretrained(base, save_dir)
      processor = AutoProcessor.from_pretrained(save_dir)
    """
    import json
    os.makedirs(save_dir, exist_ok=True)
    # Save LoRA adapters
    model.save_pretrained(save_dir)
    # Save projector weights separately (excluded from peft to avoid 4-bit crash)
    if mm_projector_path:
        try:
            base = model.base_model.model if hasattr(model, 'base_model') else model
            proj = base
            for part in mm_projector_path.split('.'): proj = getattr(proj, part)
            proj_sd = {k: v.detach().cpu().float() for k, v in proj.state_dict().items()}
            torch.save(proj_sd, os.path.join(save_dir, 'projector.pt'))
            print(f'  Projector saved → projector.pt  ({len(proj_sd)} tensors)')
        except Exception as e:
            print(f'  [WARN] Could not save projector: {e}')
    processor.save_pretrained(save_dir)
    if metadata:
        with open(os.path.join(save_dir, 'train_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    print(f'  Checkpoint → {save_dir}')


# ════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train(
    model,
    processor,
    train_dataset,
    val_dataset,
    output_dir        : str,
    mm_projector_path : str   = None,
    mode         : str   = 'qlora',
    num_epochs   : int   = 3,
    batch_size   : int   = 1,
    grad_accum   : int   = 8,
    lr           : float = 2e-4,
    max_length   : int   = 3584,
    warmup_ratio : float = 0.03,
    weight_decay : float = 0.01,
    val_every_n   : int   = 200,
    save_every_n  : int   = 500,
    log_every_n   : int   = 10,
    patience       : int   = 5,
    early_stopping : bool  = False,
    seed           : int   = 42,
    skip_baseline  : bool  = False,
):
    """
    Training loop with validation, early stopping, and checkpointing.

    ── PER-STEP FLOW ───────────────────────────────────────────────────────
    1. batch → GPU
    2. forward pass → model(input_ids, pixel_values, labels) → loss
       loss = cross-entropy averaged over answer tokens (non -100 positions)
    3. loss /= grad_accum   (scale so accumulated gradient = full-batch gradient)
    4. loss.backward()      (compute ∂loss/∂param for LoRA A,B + projector)
    5. Repeat steps 1-4 for grad_accum batches (accumulate gradients)
    6. clip_grad_norm(max=1.0)   (prevent gradient explosions)
    7. optimizer.step()          (update LoRA A,B + projector weights)
    8. scheduler.step()          (update LR)
    9. optimizer.zero_grad()     (reset gradient buffers)

    ── OPTIMIZER: AdamW ────────────────────────────────────────────────────
    Adam with weight decay. Standard for LLM fine-tuning.
    Only params with requires_grad=True are updated (LoRA + projector).
    Frozen params (vision_tower, base LLM) are completely skipped.

    ── LR SCHEDULE: Linear warmup + cosine decay ───────────────────────────
    Warmup (steps 0 → warmup_steps): LR linearly ramps 0 → peak_lr
      Prevents large gradient steps when LoRA weights are near-zero.
    Cosine decay (warmup_steps → total_steps): LR decays smoothly to ~0
      Avoids overshooting the loss minimum near the end of training.

    ── EARLY STOPPING ──────────────────────────────────────────────────────
    no_improve_count increments each time val_loss does not improve.
    Training stops when no_improve_count >= patience.
    Catches overfitting: train_loss falls but val_loss rises.
    Best checkpoint (lowest val_loss) is always preserved.
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    collator = DriveLMCollator(processor, max_length=max_length)

    # num_workers=0: safest on Colab, avoids multiprocessing issues
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collator,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collator,
        num_workers=0, pin_memory=False,
    )

    trainable    = [p for p in model.parameters() if p.requires_grad]
    n_trainable  = sum(p.numel() for p in trainable)
    print(f'\n  Trainable tensors : {len(trainable)}')
    print(f'  Trainable params  : {n_trainable/1e6:.1f}M')

    optimizer    = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    total_steps  = (len(train_loader) * num_epochs) // grad_accum
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    if mode == 'lora':
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )

    device = next(model.parameters()).device
    model.train()

    # ── Log files ────────────────────────────────────────────────────────
    # train_log.csv  — machine-readable, one row per optimizer step
    # train_log.txt  — human-readable, mirrors everything printed to console
    log_path     = os.path.join(output_dir, 'train_log.csv')
    txt_log_path = os.path.join(output_dir, 'train_log.txt')

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'global_step', 'train_loss',
            'val_loss', 'val_ppl', 'lr', 'vram_gb', 'timestamp',
        ])

    def tlog(msg: str):
        """Print to console AND append to train_log.txt simultaneously."""
        print(msg)
        with open(txt_log_path, 'a') as f:
            f.write(msg + '\n')

    # Write header to txt log
    with open(txt_log_path, 'w') as f:
        f.write(f'DriveLM Training Log\n')
        f.write(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Output : {output_dir}\n')
        f.write('=' * 55 + '\n')

    print(f'\n{"─"*55}')
    print(f'  Training config')
    print(f'{"─"*55}')
    print(f'  mode            : {mode.upper()}')
    print(f'  epochs          : {num_epochs}')
    print(f'  batch_size      : {batch_size}')
    print(f'  grad_accum      : {grad_accum}')
    print(f'  effective_batch : {batch_size * grad_accum}')
    print(f'  total_steps     : {total_steps}')
    print(f'  warmup_steps    : {warmup_steps}')
    print(f'  lr              : {lr}')
    print(f'  val_every_n     : every {val_every_n} optimizer steps')
    print(f'  early_stopping  : patience={patience} val checks')
    print(f'  max_seq_len     : {max_length}')
    print(f'{"─"*55}\n')

    # ── Verify val and save schedule ─────────────────────────────────────
    val_steps  = [s for s in range(1, total_steps+1) if s % val_every_n  == 0]
    save_steps = [s for s in range(1, total_steps+1) if s % save_every_n == 0]
    if total_steps not in save_steps:
        save_steps.append(total_steps)   # last step always saves

    print(f'  Validation will run at optimizer steps : {val_steps}')
    print(f'  Checkpoints will save at optimizer steps: {save_steps}')
    print(f'  Early stopping                          : '
          f'{"ENABLED (patience=" + str(patience) + ")" if early_stopping else "DISABLED (pass --early-stopping to enable)"}')
    print()

    # ── State ─────────────────────────────────────────────────────────────
    global_step      = 0
    best_val_loss    = float('inf')
    no_improve_count = 0
    last_val_loss    = None    # None = no validation run yet
    last_val_ppl     = None    # shown as 'pending' in progress bar
    val_has_run      = False   # flips True after first val check
    stop_training    = False

    # ── Baseline validation before any training ───────────────────────────
    if skip_baseline:
        tlog('  Baseline validation skipped (--skip-baseline).')
        tlog('  best_val_loss initialised to inf — first val check will save checkpoint.\n')
    else:
        tlog('  Running baseline validation ...')
        init_val_loss, init_val_ppl = run_validation(model, val_loader, device)
        best_val_loss = init_val_loss
        last_val_loss = init_val_loss
        last_val_ppl  = init_val_ppl
        val_has_run   = True
        tlog(f'  Baseline → val_loss={init_val_loss:.4f}  val_ppl={init_val_ppl:.2f}\n')

    # ── Epoch loop ─────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        if stop_training:
            print(f'\n  Early stopping after epoch {epoch}.')
            break

        epoch_loss  = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for local_step, batch in enumerate(pbar):
            if stop_training:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # ── Forward ──────────────────────────────────────────────────
            # Clear CUDA cache before each forward pass.
            # Prevents memory fragmentation from previous iterations
            # accumulating and causing OOM on large 6-camera samples.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # autocast wraps forward + loss in BF16 where safe.
            #
            # QLoRA: bitsandbytes already runs LLM in BF16 via
            #   bnb_4bit_compute_dtype=bfloat16, and vision tower is BF16.
            #   autocast is mostly a no-op but catches any FP32 residuals
            #   (e.g. projector LayerNorm after prepare_model_for_kbit_training
            #   casts them to FP32) and speeds them up slightly.
            #
            # LoRA: base model is FP16, LayerNorms are FP32.
            #   autocast converts eligible ops to BF16 → ~10-15% speedup.
            #
            # backward() is called OUTSIDE autocast — gradients always
            # computed in full precision for numerical stability.
            # GradScaler is NOT used because BF16 has wide enough range
            # for LLM training (unlike FP16 which needs scaling).
            try:
                outputs = model(
                    input_ids      = batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    pixel_values   = batch.get('pixel_values'),
                    labels         = batch['labels'],
                )
            except torch.cuda.OutOfMemoryError:
                print(f'\n  [OOM] step={global_step} — skipping batch')
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                continue

            loss = outputs.loss / grad_accum
            loss.backward()

            epoch_loss  += loss.item() * grad_accum
            epoch_steps += 1

            # ── Optimizer step ───────────────────────────────────────────
            if (local_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr  = scheduler.get_last_lr()[0]
                train_loss  = epoch_loss / epoch_steps
                vram_gb     = (torch.cuda.memory_allocated() / 1024**3
                               if torch.cuda.is_available() else 0.0)

                # Only show val metrics after at least one val check has run
                postfix = {
                    'tr_loss' : f'{train_loss:.4f}',
                    'lr'      : f'{current_lr:.2e}',
                    'vram'    : f'{vram_gb:.1f}G',
                }
                if val_has_run:
                    postfix['vl_loss'] = f'{last_val_loss:.4f}'
                    postfix['vl_ppl']  = f'{last_val_ppl:.2f}'
                else:
                    postfix['val'] = f'pending step {((global_step // val_every_n) + 1) * val_every_n}'
                pbar.set_postfix(postfix)

                # ── Log every optimizer step ─────────────────────────────
                # Log unconditionally — with small sample counts (50-500)
                # total_steps is small (6-62) so log_every_n=10 would miss
                # most steps. We log every step so the CSV always has a
                # complete loss curve regardless of dataset size.
                with open(log_path, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        epoch + 1, global_step,
                        round(train_loss, 6),
                        round(last_val_loss, 6) if last_val_loss is not None else '',
                        round(last_val_ppl, 4)  if last_val_ppl  is not None else '',
                        round(current_lr, 8),
                        round(vram_gb, 2),
                        time.strftime('%Y-%m-%d %H:%M:%S'),
                    ])

                # ── Periodic validation ───────────────────────────────────
                if global_step % val_every_n == 0:
                    tlog(f'\n  [step {global_step}] Validating ...')
                    val_loss, val_ppl = run_validation(model, val_loader, device)
                    last_val_loss = val_loss
                    last_val_ppl  = val_ppl
                    val_has_run   = True
                    tlog(f'  train_loss={train_loss:.4f}  '
                         f'val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}')

                    # Log again immediately after val with fresh val metrics
                    with open(log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            epoch + 1, global_step,
                            round(train_loss, 6),
                            round(val_loss, 6),
                            round(val_ppl, 4),
                            round(current_lr, 8),
                            round(vram_gb, 2),
                            time.strftime('%Y-%m-%d %H:%M:%S'),
                        ])

                    if val_loss < best_val_loss:
                        best_val_loss    = val_loss
                        no_improve_count = 0
                        save_checkpoint(model, processor,
                                        os.path.join(output_dir, 'best_checkpoint'),
                                        mm_projector_path=mm_projector_path,
                                        metadata={
                                            'epoch': epoch + 1,
                                            'global_step': global_step,
                                            'val_loss': round(val_loss, 6),
                                            'val_ppl' : round(val_ppl, 4),
                                            'train_loss': round(train_loss, 6),
                                        })
                        tlog(f'  ★ New best val_loss={best_val_loss:.4f}  '
                             f'val_ppl={val_ppl:.2f}')
                    else:
                        # Skip early stop count when val returned inf (OOM)
                        if val_loss == float('inf'):
                            print(f'  Val returned inf (OOM) — skipping early stop count.')
                            print(f'  Tip: add --img-size 224 to reduce memory usage.')
                        elif early_stopping:
                            # Only apply early stopping when explicitly requested
                            no_improve_count += 1
                            print(f'  No improvement ({no_improve_count}/{patience})')
                            if no_improve_count >= patience:
                                print(f'\n  Early stopping triggered after '
                                      f'{patience} checks without improvement.')
                                stop_training = True
                        else:
                            print(f'  No improvement (early stopping disabled — '
                                  f'pass --early-stopping to enable)')

                # ── Step checkpoint ───────────────────────────────────────
                # Also save at the very last step when save_every_n > total_steps
                is_last_step = (global_step == total_steps)
                if global_step % save_every_n == 0 or is_last_step:
                    save_checkpoint(
                        model, processor,
                        os.path.join(output_dir, f'checkpoint-{global_step}'),
                        mm_projector_path=mm_projector_path,
                        metadata={
                            'epoch': epoch + 1, 'global_step': global_step,
                            'val_loss': round(last_val_loss, 6),
                            'train_loss': round(epoch_loss / max(1, epoch_steps), 6),
                        })

        # ── End of epoch ─────────────────────────────────────────────────
        epoch_avg = epoch_loss / max(1, epoch_steps)
        tlog(f'\n  Epoch {epoch+1} done — train_loss={epoch_avg:.4f}')

        val_loss, val_ppl = run_validation(model, val_loader, device)
        last_val_loss = val_loss
        last_val_ppl  = val_ppl
        tlog(f'  Epoch {epoch+1} val   — val_loss={val_loss:.4f}  '
             f'val_ppl={val_ppl:.2f}')

        save_checkpoint(model, processor,
                        os.path.join(output_dir, f'epoch-{epoch+1}'),
                        mm_projector_path=mm_projector_path,

                        metadata={
                            'epoch': epoch + 1, 'global_step': global_step,
                            'val_loss': round(val_loss, 6),
                            'val_ppl' : round(val_ppl, 4),
                            'train_loss': round(epoch_avg, 6),
                        })

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1, global_step,
                round(epoch_avg, 6), round(val_loss, 6), round(val_ppl, 4),
                round(scheduler.get_last_lr()[0], 8),
                round(torch.cuda.memory_allocated() / 1024**3
                      if torch.cuda.is_available() else 0, 2),
                time.strftime('%Y-%m-%d %H:%M:%S'),
            ])

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            no_improve_count = 0
            save_checkpoint(model, processor,
                            os.path.join(output_dir, 'best_checkpoint'),
                            mm_projector_path=mm_projector_path,

                            metadata={
                                'epoch': epoch + 1, 'val_loss': round(val_loss, 6),
                                'val_ppl': round(val_ppl, 4),
                                'train_loss': round(epoch_avg, 6),
                            })
            tlog(f'  ★ Best checkpoint updated.')

    # ── Final ─────────────────────────────────────────────────────────────
    save_checkpoint(model, processor,
                    os.path.join(output_dir, 'final'),
                    mm_projector_path=mm_projector_path,

                    metadata={'best_val_loss': round(best_val_loss, 6),
                              'total_steps': global_step})
    tlog(f'\n{"═"*55}')
    tlog(f'  Training complete')
    tlog(f'  Best val_loss   : {best_val_loss:.4f}')
    tlog(f'  Best checkpoint : {os.path.join(output_dir, "best_checkpoint")}')
    tlog(f'  CSV log         : {log_path}')
    tlog(f'  Text log        : {txt_log_path}')
    tlog(f'{"═"*55}')


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='DriveLM LLaVA-1.5 LoRA/QLoRA Fine-tuning'
    )
    p.add_argument('--train-csv',  required=True)
    p.add_argument('--val-csv',    default=None,
                   help='If omitted, auto-splits train CSV 90/10 by category')
    p.add_argument('--images',     required=True)
    p.add_argument('--out',        default='./drivelm_checkpoints')
    p.add_argument('--mode',       choices=['lora', 'qlora'], default='qlora')
    p.add_argument('--lora-r',     type=int,   default=16)
    p.add_argument('--epochs',     type=int,   default=3)
    p.add_argument('--batch-size', type=int,   default=1)
    p.add_argument('--grad-accum', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=2e-4)
    p.add_argument('--val-every',  type=int,   default=200)
    p.add_argument('--save-every', type=int,   default=500)
    p.add_argument('--patience',      type=int,   default=5,
                   help='Early stopping patience — number of consecutive val checks '
                        'with no improvement before stopping. '
                        'Only active when --early-stopping flag is set.')
    p.add_argument('--early-stopping', action='store_true',
                   help='Enable early stopping. If NOT passed, training always '
                        'runs for the full number of epochs regardless of val loss.')
    p.add_argument('--categories', nargs='+',
                   default=['perception', 'prediction', 'planning', 'behavior'])
    p.add_argument('--max-samples',    type=int, default=None,
                   help='Limit TRAIN samples (val auto-scales to max_samples//5 '
                        'when no --val-csv provided)')
    p.add_argument('--max-val-samples', type=int, default=None,
                   help='Explicitly limit VAL samples regardless of val CSV size. '
                        'Recommended: 100-200 samples for reasonable val check time. '
                        'e.g. --max-val-samples 160')
    p.add_argument('--img-size',   type=int,   default=336)
    p.add_argument('--max-length', type=int,   default=3584)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--cpu-test',     action='store_true',
                   help='CPU smoke-test: skips quantization, runs 2 batches, '
                        'verifies full pipeline without GPU. '
                        'Use this to confirm the script works before GPU run.')
    p.add_argument('--skip-baseline', action='store_true',
                   help='Skip the baseline validation run before training starts. '
                        'Useful when val set is large (e.g. 6810 samples) and '
                        'you do not need the untrained baseline score. '
                        'Saves ~25 min on T4 with full val set.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'\n{"═"*65}')
    print(f'  DriveLM LLaVA-1.5 Fine-tuning')
    print(f'{"═"*65}')
    print(f'  Mode        : {args.mode.upper()}')
    print(f'  Train CSV   : {args.train_csv}')
    print(f'  Val CSV     : {args.val_csv or "auto-split 90/10"}')
    print(f'  Images      : {args.images}')
    print(f'  Output      : {args.out}')
    print(f'  Categories  : {args.categories}')
    print(f'  LoRA rank   : {args.lora_r}')
    print(f'  Epochs      : {args.epochs}')
    print(f'  Eff. batch  : {args.batch_size} × {args.grad_accum} = '
          f'{args.batch_size * args.grad_accum}')
    print(f'  LR          : {args.lr}')
    print(f'  Val every   : {args.val_every} steps')
    print(f'  Early stop  : {"ENABLED patience=" + str(args.patience) if args.early_stopping else "DISABLED"}')
    if args.cpu_test:
        print(f'  Device      : CPU (test mode — no GPU required)')
    elif torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU         : {props.name}  '
              f'({props.total_memory/1024**3:.1f} GB VRAM)')
    else:
        print(f'  Device      : CPU (no GPU detected)')
    print(f'{"═"*65}')

    # ── Load data ─────────────────────────────────────────────────────────
    train_df = pd.read_csv(args.train_csv)
    if args.categories:
        train_df = train_df[train_df['qa_category'].isin(args.categories)]

    if args.val_csv:
        val_df = pd.read_csv(args.val_csv)
        if args.categories:
            val_df = val_df[val_df['qa_category'].isin(args.categories)]
        print(f'\n  Val CSV: {len(val_df)} samples')
    else:
        print(f'\n  Auto-splitting 90/10 stratified by qa_category ...')
        train_df, val_df = split_dataframe(
            train_df, val_ratio=0.1, seed=args.seed
        )

    if args.max_samples:
        train_df = train_df.sample(
            min(args.max_samples, len(train_df)), random_state=args.seed
        ).reset_index(drop=True)
        # Auto-scale val only when no --val-csv provided (auto-split case)
        if not args.val_csv:
            val_df = val_df.sample(
                min(max(1, args.max_samples // 5), len(val_df)),
                random_state=args.seed
            ).reset_index(drop=True)

    # Explicit val size limit — applies to both auto-split and --val-csv
    if args.max_val_samples:
        val_df = val_df.sample(
            min(args.max_val_samples, len(val_df)), random_state=args.seed
        ).reset_index(drop=True)
        print(f'  Val limited to {len(val_df)} samples (--max-val-samples)')

    # ── Build datasets ────────────────────────────────────────────────────
    train_dataset = DriveLMDataset(train_df, args.images, args.img_size, 'train')
    val_dataset   = DriveLMDataset(val_df,   args.images, args.img_size, 'val')

    # ── CPU test mode warning ─────────────────────────────────────────────
    if args.cpu_test:
        print('\n' + '!'*65)
        print('  CPU TEST MODE — pipeline check only, no real training')
        print('  Requires ~28 GB RAM for FP32 model load.')
        print('  Use --max-samples 4 --epochs 1 to keep it fast.')
        print('  Switch to --mode qlora on GPU for real training.')
        print('!'*65)

    # ── Load model ────────────────────────────────────────────────────────
    model, processor, mm_projector_path = load_model_and_processor(
        mode     = args.mode,
        lora_r   = args.lora_r,
        cpu_test = args.cpu_test,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    train(
        model             = model,
        processor         = processor,
        train_dataset     = train_dataset,
        val_dataset       = val_dataset,
        output_dir        = args.out,
        mm_projector_path = mm_projector_path,
        mode              = args.mode,
        num_epochs        = args.epochs,
        batch_size        = args.batch_size,
        grad_accum        = args.grad_accum,
        lr                = args.lr,
        max_length        = args.max_length,
        val_every_n       = args.val_every,
        save_every_n      = args.save_every,
        patience          = args.patience,
        early_stopping    = args.early_stopping,
        seed              = args.seed,
        skip_baseline     = args.skip_baseline,
    )
