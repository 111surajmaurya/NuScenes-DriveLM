import pandas as pd
import cv2
import numpy as np
import random
import re
import argparse
import os
from PIL import Image, ImageDraw, ImageFont

# -------------------------------
# ARGUMENTS
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True, help="Path to QA CSV")
parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to visualize")
parser.add_argument("--img_root", type=str, default="data", help="Root path for images")
parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

IMG_SIZE = (400, 300)

# -------------------------------
# HELPERS
# -------------------------------

CAM_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"
]

CAM_LABEL = {
    "CAM_FRONT": "FRONT",
    "CAM_FRONT_LEFT": "FRONT_LEFT",
    "CAM_FRONT_RIGHT": "FRONT_RIGHT",
    "CAM_BACK": "BACK",
    "CAM_BACK_LEFT": "BACK_LEFT",
    "CAM_BACK_RIGHT": "BACK_RIGHT",
}


def parse_image_paths(path_str):
    result = {}
    parts = path_str.split("|")
    for p in parts:
        if ":" in p:
            cam, path = p.strip().split(":", 1)
            result[cam.strip()] = path.strip()
    return result


def parse_bbox_from_text(ref_text):
    bboxes = []
    matches = re.findall(r"bbox:\[([0-9., ]+)\]", ref_text)
    for m in matches:
        nums = [float(x) for x in m.split(",")]
        if len(nums) == 4:
            bboxes.append(nums)
    return bboxes


def draw_boxes(img, bboxes):
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def put_label(img, text):
    cv2.putText(
        img,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,  # slightly bigger
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def load_image(path):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img


def wrap_text(text, max_chars=80):
    """Simple word wrap"""
    words = text.split()
    lines = []
    current = ""

    for w in words:
        if len(current) + len(w) < max_chars:
            current += " " + w
        else:
            lines.append(current.strip())
            current = w

    if current:
        lines.append(current.strip())

    return "\n".join(lines)


# -------------------------------
# MAIN
# -------------------------------

df = pd.read_csv(args.csv)

samples = df.sample(min(args.num_samples, len(df)))

for idx, (_, row) in enumerate(samples.iterrows()):

    print(f"\nSample {idx}")
    print("Q:", row["question_readable"])
    print("A:", row["answer_readable"])

    img_dict = parse_image_paths(row["all_image_paths"])
    bboxes = parse_bbox_from_text(str(row["referenced_objects"]))

    images = {}

    for cam in CAM_ORDER:
        path = img_dict.get(cam)

        if path:
            path = path.replace("..", args.img_root)
            img = load_image(path)

            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                img = draw_boxes(img, bboxes)
                img = put_label(img, CAM_LABEL[cam])
            else:
                img = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
        else:
            img = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)

        images[cam] = img

    # -------------------------------
    # GRID
    # -------------------------------

    top_row = np.hstack([
        images["CAM_FRONT_LEFT"],
        images["CAM_FRONT"],
        images["CAM_FRONT_RIGHT"]
    ])

    bottom_row = np.hstack([
        images["CAM_BACK_LEFT"],
        images["CAM_BACK"],
        images["CAM_BACK_RIGHT"]
    ])

    grid = np.vstack([top_row, bottom_row])

    # -------------------------------
    # TEXT (LARGER + WRAPPED)
    # -------------------------------

    grid_pil = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))

    q_text = wrap_text("Q: " + str(row["question_readable"]), 90)
    a_text = wrap_text("A: " + str(row["answer_readable"]), 90)

    full_text = q_text + "\n\n" + a_text

    text_height = 180  # increased
    new_img = Image.new("RGB", (grid_pil.width, grid_pil.height + text_height), (0, 0, 0))
    new_img.paste(grid_pil, (0, 0))

    draw = ImageDraw.Draw(new_img)

    try:
        font = ImageFont.truetype("arial.ttf", 28)  
    except:
        font = ImageFont.load_default()

    draw.text((10, grid_pil.height + 10), full_text, fill=(255, 255, 255), font=font)

    # -------------------------------
    # SAVE
    # -------------------------------

    out_path = os.path.join(args.out_dir, f"sample_{idx}.jpg")
    new_img.save(out_path)

    print(f"Saved: {out_path}")