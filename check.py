import sys
import importlib

def check(pkg):
    try:
        module = importlib.import_module(pkg)
        version = getattr(module, "__version__", "unknown")
        print(f"{pkg:<20} ✅ {version}")
    except Exception as e:
        print(f"{pkg:<20} ❌ NOT INSTALLED ({e})")

print("\n===== PYTHON INFO =====")
print("Python:", sys.version)

print("\n===== CORE PACKAGES =====")
packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "tokenizers",
    "accelerate",
    "bitsandbytes",
    "sentencepiece",
    "huggingface_hub",
    "PIL",
    "cv2",
]

for p in packages:
    check(p)

print("\n===== TORCH DETAILS =====")
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch error:", e)

print("\n===== TRANSFORMERS CHECK =====")
try:
    from transformers import LlavaProcessor
    print("LlavaProcessor import ✅")
except Exception as e:
    print("LlavaProcessor import ❌", e)

print("\n===== TOKENIZER TEST =====")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        use_fast=False
    )
    print("Tokenizer load ✅")
except Exception as e:
    print("Tokenizer load ❌", e)

# print("\n===== PROCESSOR TEST =====")
# try:
#     from transformers import AutoProcessor
#     hf_id = "llava-hf/llava-1.5-7b-hf"
#     proc = AutoProcessor.from_pretrained(
#         hf_id,
#         force_download=True
#     )
#     print("Processor load ✅")
# except Exception as e:
#     print("Processor load ❌", e)

print("\n===== MODEL TEST =====")
try:
    import torch
    from transformers import LlavaForConditionalGeneration

    hf_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model load ✅")
except Exception as e:
    print("Model load ❌", e)

print("\n===== DONE =====")