# DriveLM + nuScenes 

This project parses **DriveLM QA data** aligned with **nuScenes**, enriches it with metadata, performs basic analysis, and provides visualization utilities.

---

## 1. Setup (Docker)

Build the Docker image:

```bash
sudo docker build -t nuscene .
```

Run the container:

```bash
docker run -it --gpus all \
  -p 8888:8888 \
  -v ~/workspace/assignment:/workspace/assignment \
  -v ~/workspace/assignment/data:/data \
  nuscene
```

---

## 2. Launch Jupyter Notebook

Inside the container:

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open in browser:

http://localhost:8888

---

## 3. Data Parsing + Analysis

Run the parser:

```bash
python scripts/parse_drivelm.py ./data/QA_dataset_nus/v1_0_train_nus.json
```

### Output

This generates a folder:

drivelm_parsed/
├── qa_enriched.csv      # Main dataset (Q-A pairs)
├── objects.csv
├── frames.csv
├── scenes.csv
└── analysis.txt         # Analysis summary

### Notes

- Uses:
  - 6 overlapping scenes (DriveLM + nuScenes mini)
  - + 9 additional DriveLM scenes
- Total processed scenes: 15
- qa_enriched.csv is the main file for modeling and analysis.

---

## 4. Visualization

Run:

```bash
python scripts/vis_data.py \
  --csv ./drivelm_parsed/qa_enriched.csv \
  --num_sample 20
```


### Output

outputs/
├── sample_1.jpg
├── sample_2.jpg
...

Each visualization includes:

- 6-camera layout:
  FRONT_LEFT | FRONT | FRONT_RIGHT
  BACK_LEFT  | BACK  | BACK_RIGHT

- Bounding boxes for referenced objects
- Question and Answer overlay


## Split train-val dataset

```
python3 scripts/benchmark_local.py     --csv ./drivelm_parsed/drivelm_splits/val/qa_enriched.csv     --images ./data/nuscenes     --limit 10
```

---

## 5. Features

- DriveLM QA parsing
- nuScenes metadata integration
- Object-level grounding (bbox + camera)
- QA enrichment:
  - object references
  - camera zones (front/back/both)
  - question types
- Dataset analysis:
  - distributions
  - bias & gap detection
- Multi-camera visualization

---

## 6. Project Structure

.
├── data/
├── scripts/
│   ├── parse_drivelm2.py
│   └── vis_data.py
├── drivelm_parsed/
├── outputs/
├── Dockerfile
└── README.md

---

## 7. Tips

- If images don’t load:
  - Check path replacement (`..` → `data`)
- Use smaller samples for quick debugging:
  --num_sample 5
- Use pandas for deeper analysis on qa_enriched.csv

---

## 8. Future Improvements

- Object labels on bounding boxes  
- Color-coded objects  
- Multi-frame (temporal) visualization  
- Video playback support  

---

## 9. Quick Summary

| Step | Command |
|------|--------|
| Build Docker | sudo docker build -t nuscene . |
| Run container | docker run ... |
| Start Jupyter | jupyter notebook ... |
| Parse data | python scripts/parse_drivelm2.py ... |
| Visualize | python scripts/vis_data.py ... |

---

## License

Internal / Research use (update as needed)
