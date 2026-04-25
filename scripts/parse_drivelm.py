"""
DriveLM + nuScenes  —  Parser + Analysis  (v3)
===============================================
Verified logic:
  - obj refs use semicolons (commas replaced by safe())
  - camera extraction from both ref tokens and image_path strings
  - question type classification with regex
  - camera zone (front/back/both/none) per QA
  - answer/question word count
  - object count per QA

Run:
  python3 parse_drivelm_v3.py  <drivelm_json> 

Outputs  (./drivelm_parsed/):
  qa_enriched.csv    — one row per QA (primary table)
  objects.csv        — one row per object per frame
  frames.csv         — one row per key frame
  scenes.csv         — one row per scene
  analysis/

"""

import json, re, os, random, csv
import pandas as pd
from pathlib import Path
from collections import Counter
from nuscenes.nuscenes import NuScenes


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

CAMERAS = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
           'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']

CAMERA_MAPPING = {
    'CAM_FRONT'      : 'front camera',
    'CAM_FRONT_LEFT' : 'front-left camera',
    'CAM_FRONT_RIGHT': 'front-right camera',
    'CAM_BACK'       : 'back camera',
    'CAM_BACK_LEFT'  : 'back-left camera',
    'CAM_BACK_RIGHT' : 'back-right camera',
}

FRONT_CAMS = {'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'}
BACK_CAMS  = {'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT'}

# Regex patterns — AFTER safe() replaces commas with semicolons
OBJ_REF_PATTERN = re.compile(r'<c\d+;([A-Z_]+);[\d.]+;[\d.]+>')
CAM_IN_PATH_PATTERN = re.compile(r'(CAM_[A-Z_]+):')


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def safe(text) -> str:
    """Sanitize for CSV: replace commas and newlines."""
    if text is None or (isinstance(text, float)):
        return ''
    return str(text).replace(',', ';').replace('\n', ' ').replace('\r', '').strip()


def parse_object_key(obj_key: str) -> dict:
    """Parse '<c1,CAM_FRONT,1300.8,792.5>' into components."""
    inner = obj_key.strip('<>').split(',')
    return {
        'obj_id'  : obj_key,
        'obj_idx' : inner[0],
        'camera'  : inner[1],
        'center_x': float(inner[2]),
        'center_y': float(inner[3]),
    }


def build_object_lookup(key_object_infos: dict) -> dict:
    """Map obj_key → human-readable string."""
    lookup = {}
    for obj_key, info in key_object_infos.items():
        parsed    = parse_object_key(obj_key)
        cam_human = CAMERA_MAPPING.get(parsed['camera'], parsed['camera'])
        category  = info.get('Category') or ''
        status    = info.get('Status')   or ''
        desc      = (info.get('Visual_description') or '').rstrip('.')
        human = f"{status} {desc} in {cam_human}".strip() if status \
                else f"{desc} ({category}) in {cam_human}"
        lookup[obj_key] = human
    return lookup


def replace_object_refs(text: str, lookup: dict) -> str:
    """Replace <cN,CAM_*,x,y> tokens with human descriptions."""
    if not text:
        return text
    pattern = re.compile(r'<c\d+,[A-Z_]+,[\d.]+,[\d.]+>')
    def replacer(match):
        key = match.group(0)
        return f'{lookup[key]}' if key in lookup else f'{key}'
    return pattern.sub(replacer, text)


def count_obj_refs(ref_str) -> int:
    """Count object references in a (post-safe) ref string."""
    if not ref_str or str(ref_str) in ('nan', ''):
        return 0
    return len(re.findall(r'<c\d+;[A-Z_]+;[\d.]+;[\d.]+>', str(ref_str)))


def extract_cameras_from_refs(ref_str) -> list:
    """Extract camera names from ref string: '<c2;CAM_BACK_RIGHT;...'"""
    if not ref_str or str(ref_str) in ('nan', ''):
        return []
    return OBJ_REF_PATTERN.findall(str(ref_str))


def extract_cameras_from_imgpaths(path_str) -> list:
    """Extract camera names from 'CAM_FRONT:path | CAM_BACK:path' string."""
    if not path_str or str(path_str) == 'nan':
        return []
    return CAM_IN_PATH_PATTERN.findall(str(path_str))


def camera_zone(cams: list) -> str:
    """Classify camera list as front_only / back_only / both / none."""
    s = set(cams)
    has_front = bool(s & FRONT_CAMS)
    has_back  = bool(s & BACK_CAMS)
    if has_front and has_back: return 'both'
    if has_front: return 'front_only'
    if has_back:  return 'back_only'
    return 'none'


def classify_question(q: str) -> str:
    """Rule-based question type classification."""
    q = str(q).lower().strip()
    if q.startswith('what is the status'):       return 'status_query'
    if q.startswith('what is the visual'):        return 'visual_description'
    if q.startswith('what is the color'):         return 'color_query'
    if q.startswith('what is the speed'):         return 'speed_query'
    if q.startswith('what is the distance'):      return 'distance_query'
    if q.startswith('what is the direction'):     return 'direction_query'
    if q.startswith('what is the behavior'):      return 'behavior_query'
    if q.startswith('what are the objects'):      return 'object_enumeration'
    if q.startswith('what are objects'):          return 'object_enumeration'
    if q.startswith('what'):                      return 'what_query'
    if q.startswith('predict the behavior'):      return 'ego_behavior_prediction'
    if q.startswith('predict'):                   return 'prediction_other'
    if q.startswith('will '):                     return 'future_state'
    if q.startswith('is ') or q.startswith('are '): return 'yes_no'
    if q.startswith('how many'):                  return 'counting'
    if q.startswith('how'):                       return 'how_other'
    if 'should' in q and 'consider' in q:         return 'planning_priority'
    if 'should' in q:                             return 'planning_other'
    if q.startswith('which'):                     return 'which_query'
    if q.startswith('where'):                     return 'location_query'
    if q.startswith('why'):                       return 'reasoning'
    return 'other_question_type'

def parse_nuscene(dataroot: str = "./data/nuscenes_v1.0"):
    """
    Initialize NuScenes devkit (mini split).

    Returns:
        NuScenes object with metadata (scenes, samples, etc.)
    """
    return NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)


def parse_drivelm(json_path: str) -> dict[str, pd.DataFrame]:
    """
    Parse DriveLM dataset and align with NuScenes-mini.

    Steps:
    1. Load NuScenes-mini and extract scene tokens
    2. Load DriveLM JSON
    3. Select:
        - all common scenes between NuScenes-mini & DriveLM
        - +9 additional scenes from DriveLM (to reach 15 total)
    4. Parse into structured tables:
        scenes, frames, objects, QA

    Returns:
        dict of pandas DataFrames
    """

    # ─────────────────────────────────────────────────────────────
    # Load NuScenes (only for scene filtering, no heavy usage)
    # ─────────────────────────────────────────────────────────────
    nusc = parse_nuscene()
    mini_scene_tokens = {scene["token"] for scene in nusc.scene}

    # ─────────────────────────────────────────────────────────────
    # Load DriveLM
    # ─────────────────────────────────────────────────────────────
    with open(json_path, "r") as f:
        drivelm = json.load(f)

    drivelm_scene_tokens = list(drivelm.keys())

    # ─────────────────────────────────────────────────────────────
    # Select scenes (6 common + 9 extra)
    # ─────────────────────────────────────────────────────────────
    common_scenes = list(set(drivelm_scene_tokens) & mini_scene_tokens)
    print(f"Common scenes (NuScenes-mini ∩ DriveLM): {len(common_scenes)}")

    # pick extra scenes from DriveLM (excluding common ones)
    remaining_scenes = [s for s in drivelm_scene_tokens if s not in common_scenes]

    # deterministic selection (important for reproducibility)
    remaining_scenes = sorted(remaining_scenes)

    extra_needed = max(0, 15 - len(common_scenes))
    extra_scenes = remaining_scenes[:extra_needed]

    selected_scenes = common_scenes + extra_scenes

    print(f"Total selected scenes: {len(selected_scenes)} (common + extra)")

    # filter data
    data = {scene: drivelm[scene] for scene in selected_scenes}

    # ─────────────────────────────────────────────────────────────
    # Storage
    # ─────────────────────────────────────────────────────────────
    scenes_rows  = []
    frames_rows  = []
    objects_rows = []
    qa_rows      = []

    # ─────────────────────────────────────────────────────────────
    # Parsing loop
    # ─────────────────────────────────────────────────────────────
    for scene_token, scene_data in data.items():

        scene_desc = scene_data.get("scene_description", "")
        key_frames = scene_data.get("key_frames", {})

        # ── Scene level ───────────────────────────────────────────
        scenes_rows.append({
            "scene_token": scene_token,
            "scene_description": scene_desc,
            "num_key_frames": len(key_frames),
        })

        for frame_token, frame_data in key_frames.items():

            key_object_infos = frame_data.get("key_object_infos", {})
            qa_data          = frame_data.get("QA", {})
            image_paths      = frame_data.get("image_paths", {})

            # build object lookup for replacement
            obj_lookup = build_object_lookup(key_object_infos)

            # serialize image paths
            img_paths_str = " | ".join(
                f"{cam}:{path}" for cam, path in image_paths.items()
            )

            # object summary (for debugging / QA context)
            objects_summary = " | ".join(
                f"{obj_lookup[k]} [bbox:{v.get('2d_bbox')}]"
                for k, v in key_object_infos.items()
            )

            # ── Frame level ────────────────────────────────────────
            frames_rows.append({
                "scene_token": scene_token,
                "frame_token": frame_token,
                "scene_description": scene_desc,
                "num_objects": len(key_object_infos),
                "num_cameras": len(image_paths),
                "qa_categories": ", ".join(qa_data.keys()),
                'num_perception_qa' : len(qa_data.get('perception', [])),
                'num_prediction_qa' : len(qa_data.get('prediction', [])),
                'num_planning_qa'   : len(qa_data.get('planning',   [])),
                'num_behavior_qa'   : len(qa_data.get('behavior',   [])),
                'total_qa'          : sum(len(v) for v in qa_data.values()),
                "image_paths": img_paths_str,
            })

            # ── Object level ───────────────────────────────────────
            for obj_key, obj_info in key_object_infos.items():

                parsed = parse_object_key(obj_key)
                bbox   = obj_info.get("2d_bbox", [None]*4)

                objects_rows.append({
                    "scene_token": scene_token,
                    "frame_token": frame_token,
                    "obj_id": parsed["obj_id"],
                    "obj_idx": parsed["obj_idx"],
                    "camera": parsed["camera"],
                    "CAMERA_MAPPING": CAMERA_MAPPING.get(parsed["camera"], parsed["camera"]),
                    "center_x": parsed["center_x"],
                    "center_y": parsed["center_y"],
                    "category": obj_info.get("Category"),
                    "status": obj_info.get("Status"),
                    "visual_description": obj_info.get("Visual_description"),
                    "human_readable": obj_lookup[obj_key],
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "bbox_width": (bbox[2] - bbox[0]) if None not in bbox else None,
                    "bbox_height": (bbox[3] - bbox[1]) if None not in bbox else None,
                    "image_path": image_paths.get(parsed["camera"], ""),
                })

            # ── QA level ───────────────────────────────────────────
            for qa_category, qa_list in qa_data.items():
                for qa_idx, qa in enumerate(qa_list):

                    q_raw = qa.get("Q", "")
                    a_raw = qa.get("A", "")

                    q_replaced = replace_object_refs(q_raw, obj_lookup)
                    a_replaced = replace_object_refs(a_raw, obj_lookup)

                    # extract object references
                    obj_refs_q = re.findall(r"<c\d+,[A-Z_]+,[\d.]+,[\d.]+>", q_raw)
                    obj_refs_a = re.findall(r"<c\d+,[A-Z_]+,[\d.]+,[\d.]+>", a_raw)
                    all_refs   = list(dict.fromkeys(obj_refs_q + obj_refs_a))

                    # relevant images (based on object cameras)
                    if all_refs:
                        ref_cams = list(dict.fromkeys(
                            parse_object_key(r)["camera"] for r in all_refs
                        ))
                        relevant_imgs = {
                            cam: image_paths[cam]
                            for cam in ref_cams if cam in image_paths
                        }
                    else:
                        relevant_imgs = image_paths

                    relevant_imgs_str = " | ".join(
                        f"{cam}:{path}" for cam, path in relevant_imgs.items()
                    )

                    referenced_objects_detail = " | ".join(
                        f"{obj_lookup.get(r, r)} → bbox:{key_object_infos.get(r, {}).get('2d_bbox')}"
                        for r in all_refs
                    )

                    # Derived analysis fields
                    n_refs_q    = len(obj_refs_q)
                    n_refs_a    = len(obj_refs_a)
                    n_refs_total= n_refs_q + n_refs_a
                    rel_cams    = extract_cameras_from_imgpaths(relevant_imgs_str)
                    cam_zone    = camera_zone(rel_cams)
                    q_type      = classify_question(q_raw)
                    q_words     = len(str(q_raw).split())
                    a_words     = len(str(a_raw).split())


                    qa_rows.append({
                        "scene_token": scene_token,
                        "frame_token": frame_token,
                        "qa_category": qa_category,
                        "qa_idx": qa_idx,
                        
                        "question_raw": q_raw,
                        "question_readable": q_replaced,
                        "answer_raw": a_raw,
                        "answer_readable": a_replaced,

                        "scene_description": scene_desc,
                        "relevant_image_paths": relevant_imgs_str,
                        "all_image_paths": img_paths_str,
                        # "all_objects": objects_summary,
                        "obj_refs_in_q": ", ".join(obj_refs_q),
                        "obj_refs_in_a": ", ".join(obj_refs_a),
                        "referenced_objects": referenced_objects_detail,

                        "relevant_image_paths": relevant_imgs_str,

                        "question_type"       : q_type,
                        "n_obj_refs_in_q"     : n_refs_q,
                        "n_obj_refs_in_a"     : n_refs_a,
                        "n_obj_refs_total"    : n_refs_total,
                        "relevant_cameras"    : '; '.join(rel_cams),
                        "camera_zone"         : cam_zone,
                        "is_front_camera_qa"  : bool(set(rel_cams) & FRONT_CAMS),
                        "is_back_camera_qa"   : bool(set(rel_cams) & BACK_CAMS),
                        "question_word_count" : q_words,
                        "answer_word_count"   : a_words,

                        "choices": qa.get("C"),
                        "con_up": str(qa.get("con_up")) if qa.get("con_up") else None,
                        "con_down": str(qa.get("con_down")) if qa.get("con_down") else None,
                        "cluster": qa.get("cluster"),
                        "layer": qa.get("layer"),
                    })

    return {
        "scenes": pd.DataFrame(scenes_rows),
        "frames": pd.DataFrame(frames_rows),
        "objects": pd.DataFrame(objects_rows),
        "qa": pd.DataFrame(qa_rows),
    }


# ════════════════════════════════════════════════════════════════════════════
# PART 3 — Analysis
# ════════════════════════════════════════════════════════════════════════════

def run_analysis(qa: pd.DataFrame, obj: pd.DataFrame,
                 frames: pd.DataFrame, scenes: pd.DataFrame,
                 output_dir: str):

    ana_dir = os.path.join(output_dir, 'analysis')
    output_path = os.path.join(output_dir, "analysis.txt")
    os.makedirs(ana_dir, exist_ok=True)

    def save(df, name):
        path = os.path.join(ana_dir, name)
        df.to_csv(path, index=True, quoting=1)
        return path

    lines = []   # for summary txt
    bias  = []   # for bias report

    def section(title):

        sep = '─' * 60
        lines.append(f'\n{sep}\n  {title}\n{sep}')
        print(f'\n  ── {title} ──')

    total_qa    = len(qa)
    total_frames= len(frames)
    total_scenes= len(scenes)
    total_obj   = len(obj)

    lines.append('═'*60)
    lines.append('  DriveLM Dataset Analysis Report')
    lines.append('═'*60)
    lines.append(f'  Total scenes      : {total_scenes}')
    lines.append(f'  Total key frames  : {total_frames}')
    lines.append(f'  Total QA pairs    : {total_qa}')
    lines.append(f'  Total objects     : {total_obj}')

    # ── 01  QA Category Frequency ─────────────────────────────────────────
    section('01  QA Category Frequency')
    cat_freq = qa['qa_category'].value_counts().rename_axis('qa_category').reset_index(name='count')
    cat_freq['pct'] = (cat_freq['count'] / total_qa * 100).round(1)
    save(cat_freq, '01_qa_category_freq.csv')
    lines.append(cat_freq.to_string(index=False))
    print(cat_freq.to_string(index=False))

    dominant = cat_freq.iloc[0]['qa_category']
    dominant_pct = cat_freq.iloc[0]['pct']
    if dominant_pct > 60:
        bias.append(f'IMBALANCE: "{dominant}" is {dominant_pct}% of all QAs — heavily skewed.')

    # ── 02  Question Type Frequency ───────────────────────────────────────
    section('02  Question Type Frequency (fine-grained)')
    qtype_freq = qa['question_type'].value_counts().rename_axis('question_type').reset_index(name='count')
    qtype_freq['pct'] = (qtype_freq['count'] / total_qa * 100).round(1)
    save(qtype_freq, '02_question_type_freq.csv')
    lines.append(qtype_freq.to_string(index=False))
    print(qtype_freq.to_string(index=False))

    if 'reasoning' not in qtype_freq['question_type'].values or \
       qtype_freq[qtype_freq['question_type']=='reasoning']['count'].sum() < total_qa * 0.05:
        bias.append('GAP: "Why/reasoning" questions are rare (<5%) — dataset lacks causal reasoning.')
    if 'counting' not in qtype_freq['question_type'].values:
        bias.append('GAP: No "how many" counting questions detected.')

    # ── 03  Camera Mention Frequency ──────────────────────────────────────
    section('03  Individual Camera Mention Frequency')
    cam_counts = Counter()
    for cams in qa['relevant_cameras'].dropna():
        for cam in str(cams).split('; '):
            if cam.strip():
                cam_counts[cam.strip()] += 1
    cam_df = pd.DataFrame(cam_counts.most_common(), columns=['camera', 'mentions'])
    cam_df['pct'] = (cam_df['mentions'] / cam_df['mentions'].sum() * 100).round(1)
    save(cam_df, '04_camera_mention_freq.csv')
    lines.append(cam_df.to_string(index=False))
    print(cam_df.to_string(index=False))

    if len(cam_df) > 0:
        top_cam = cam_df.iloc[0]['camera']
        top_pct = cam_df.iloc[0]['pct']
        if top_pct > 40:
            bias.append(f'BIAS: {top_cam} appears in {top_pct}% of camera mentions — other views underused.')

    # ── 04  Number of Object Refs per QA ──────────────────────────────────
    section('04  Object References per QA Distribution')
    ref_dist = qa['n_obj_refs_total'].value_counts().sort_index().rename_axis('n_objects_referenced').reset_index(name='count')
    ref_dist['pct'] = (ref_dist['count'] / total_qa * 100).round(1)
    save(ref_dist, '05_objects_per_qa_dist.csv')
    lines.append(ref_dist.to_string(index=False))
    print(ref_dist.to_string(index=False))

    zero_ref_pct = qa['n_obj_refs_total'].eq(0).mean() * 100
    lines.append(f'  QAs with no specific object refs: {zero_ref_pct:.1f}%')
    if zero_ref_pct > 70:
        bias.append(f'GAP: {zero_ref_pct:.1f}% of QAs have no specific object references — mostly generic scene-level questions.')

    # ── 05  Answer Length Distribution ────────────────────────────────────
    section('05  Answer Word Count Statistics')
    a_stats = qa['answer_word_count'].describe().round(2).to_frame('answer_words')
    save(a_stats, '06_answer_length_stats.csv')
    lines.append(a_stats.to_string())
    print(a_stats.to_string())

    # Percentile bins
    bins   = [0, 5, 10, 20, 50, 999]
    labels = ['1-5', '6-10', '11-20', '21-50', '50+']
    qa['answer_length_bin'] = pd.cut(qa['answer_word_count'], bins=bins, labels=labels)
    bin_dist = qa['answer_length_bin'].value_counts().sort_index().reset_index()
    bin_dist.columns = ['answer_length_bin', 'count']
    lines.append('\n  Answer length buckets:\n' + bin_dist.to_string(index=False))

    # ── 06  Question Length Distribution ──────────────────────────────────
    section('06  Question Word Count Statistics')
    q_stats = qa['question_word_count'].describe().round(2).to_frame('question_words')
    save(q_stats, '07_question_length_stats.csv')
    lines.append(q_stats.to_string())
    print(q_stats.to_string())

    # ── 08  Object Category Frequency ────────────────────────────────────
    section('08  Object Category Frequency')
    obj_cat = obj['category'].value_counts().rename_axis('category').reset_index(name='count')
    obj_cat['pct'] = (obj_cat['count'] / total_obj * 100).round(1)
    save(obj_cat, '09_object_category_freq.csv')
    lines.append(obj_cat.to_string(index=False))
    print(obj_cat.to_string(index=False))

    # ── 09  Object Status Frequency ───────────────────────────────────────
    section('09  Object Status Frequency')
    obj_status = obj['status'].fillna('Unknown').value_counts().rename_axis('status').reset_index(name='count')
    obj_status['pct'] = (obj_status['count'] / total_obj * 100).round(1)
    save(obj_status, '10_object_status_freq.csv')
    lines.append(obj_status.to_string(index=False))
    print(obj_status.to_string(index=False))

    stationary_pct = (obj['status'] == 'Stationary').sum() / total_obj * 100
    if stationary_pct > 60:
        bias.append(f'BIAS: {stationary_pct:.1f}% of annotated objects are Stationary — dynamic object scenarios underrepresented.')

    # ── 10  Objects per Frame ─────────────────────────────────────────────
    section('10  Objects per Frame Statistics')
    obj_per_frame = obj.groupby('frame_token').size().describe().round(2).to_frame('objects_per_frame')
    save(obj_per_frame, '11_objects_per_frame_stats.csv')
    lines.append(obj_per_frame.to_string())
    print(obj_per_frame.to_string())

    # Objects per camera
    obj_cam = obj['camera'].value_counts().rename_axis('camera').reset_index(name='object_count')
    obj_cam['pct'] = (obj_cam['object_count'] / total_obj * 100).round(1)
    lines.append('\n  Objects per camera:\n' + obj_cam.to_string(index=False))
    print('\n  Objects per camera:')
    print(obj_cam.to_string(index=False))

    # ── 11  QA per Frame Distribution ─────────────────────────────────────
    section('11  QA Pairs per Frame Distribution')
    qa_per_frame = qa.groupby('frame_token').size().describe().round(2).to_frame('qa_per_frame')
    lines.append(qa_per_frame.to_string())
    print(qa_per_frame.to_string())

    # # ── 12  QA Category per Frame (are all frames balanced?) ──────────────
    # section('12  QA Category Coverage per Frame')
    # frame_cats = qa.groupby('frame_token')['qa_category'].apply(
    #     lambda x: '; '.join(sorted(x.unique()))
    # ).reset_index(name='categories_present')
    # cat_coverage = frame_cats['categories_present'].value_counts().reset_index()
    # cat_coverage.columns = ['categories_present', 'num_frames']
    # lines.append(cat_coverage.to_string(index=False))
    # print(cat_coverage.to_string(index=False))

    # frames_missing_planning = frames[frames['num_planning_qa'] == 0]
    # frames_missing_pred     = frames[frames['num_prediction_qa'] == 0]
    # lines.append(f'  Frames with 0 planning QAs   : {len(frames_missing_planning)} / {total_frames}')
    # lines.append(f'  Frames with 0 prediction QAs : {len(frames_missing_pred)} / {total_frames}')
    # if len(frames_missing_planning) / total_frames > 0.5:
    #     bias.append(f'GAP: {len(frames_missing_planning)}/{total_frames} frames have NO planning QAs.')
    # if len(frames_missing_pred) / total_frames > 0.5:
    #     bias.append(f'GAP: {len(frames_missing_pred)}/{total_frames} frames have NO prediction QAs.')

    # ── Bias & Gap Report ─────────────────────────────────────────────────
    section('BIASES & GAPS DETECTED')
    if bias:
        for b in bias:
            lines.append(f'{b}')
            print(f'{b}')
    else:
        lines.append('No obvious biases detected in this sample.')
        print('No obvious biases detected in this sample.')

    # Save reports
    summary_path = os.path.join(ana_dir, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))

    bias_path = os.path.join(ana_dir, '12_bias_gaps_report.txt')
    with open(bias_path, 'w') as f:
        f.write('DriveLM Bias & Gap Report\n')
        f.write('='*50 + '\n')
        if bias:
            for b in bias:
                f.write(f'⚠️  {b}\n')
        else:
            f.write('No obvious biases detected.\n')

    print(f'\nAnalysis saved to {ana_dir}/')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved analysis to: {output_path}")
    
    return ana_dir


# ════════════════════════════════════════════════════════════════════════════
# SAVE CSVs
# ════════════════════════════════════════════════════════════════════════════

def save_csvs(dfs: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for name, df in dfs.items():
        path = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
        print(f'{name}.csv  →  {len(df):,} rows  |  {len(df.columns)} cols')


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, csv as csv_module

    drivelm_json  = sys.argv[1] if len(sys.argv) > 1 else 'v1_1_train_nus.json'
    output_dir    = sys.argv[4] if len(sys.argv) > 4 else './drivelm_parsed'

    print(f'\n{"═"*65}')
    print(f'  DriveLM JSON  : {drivelm_json}')
    print(f'  Output dir    : {output_dir}')
    print(f'{"═"*65}\n')

    print('Step 1: Parsing DriveLM ...')
    dfs = parse_drivelm(drivelm_json)
    print(f'  Scenes: {len(dfs["scenes"])}  |  Frames: {len(dfs["frames"])}  |  '
          f'Objects: {len(dfs["objects"])}  |  QA: {len(dfs["qa"])}')


    print(f'\nSaving CSVs ...')
    all_dfs = {
        'qa_enriched'  : dfs['qa'],
        'objects'      : dfs['objects'],
        'frames'       : dfs['frames'],
        'scenes'       : dfs['scenes'],
    }
    save_csvs(all_dfs, output_dir)

    # Step 8 — Analysis
    print(f'\nStep 5: Running analysis ...')
    run_analysis(
        qa      = dfs['qa'],       # use full QA (not just overlap) for stats
        obj     = dfs['objects'],
        frames  = dfs['frames'],
        scenes  = dfs['scenes'],
        output_dir = output_dir,
    )
