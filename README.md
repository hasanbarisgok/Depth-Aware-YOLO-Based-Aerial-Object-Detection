# Depth-Aware YOLO-Based Aerial Object Detection

This repository contains the final project structure for a YOLOv8-based aerial object detection pipeline with pseudo-depth visualization support. The system detects four aerial object classes and exposes both script-based inference and an API layer for frontend integration.

## Live Demo

The system is currently available online at:

- [https://hasanbarisgok.com/aod4-demo](https://hasanbarisgok.com/aod4-demo)

## Academic Context

**Course:** Computer Vision (`CENG0038`)  
**Instructor:** Assoc. Prof. Dr. Serkan KARTAL  
**Instructor Website:** [ceng.cu.edu.tr/skartal](https://ceng.cu.edu.tr/skartal/index.html)  
**Instructor LinkedIn:** [serkan-kartal-39063b14a](https://www.linkedin.com/in/serkan-kartal-39063b14a/?locale=tr)

**Project Team**

- Hasan Barış GÖK  
  M.Sc. Student, Department of Computer Engineering, Çukurova University  
  AI-Augmented Frontend Engineer @ Sunflower Care e.V.  
  Email: `hasanbarisgok@gmail.com`  
  Website: [hasanbarisgok.com](https://hasanbarisgok.com)

- Halit AKCA  
  M.Sc. Student, Department of Computer Engineering, Çukurova University  
  Research Assistant @ ATU, Department of Artificial Intelligence Engineering  
  Email: `hakca@atu.edu.tr`  
  LinkedIn: [halit-akca-878072224](https://www.linkedin.com/in/halit-akca-878072224/)

## Project Scope

- YOLOv8-based aerial object detection
- Single-image inference
- Pseudo-depth map generation
- Combined detection + depth demo
- FastAPI-based inference service
- Final trained model and test artifacts

## Detected Classes

- airplane
- bird
- drone
- helicopter

## Repository Structure

```text
AOD4_GitHub/
├── configs/
│   └── data.yaml
├── models/
│   ├── aod4_total50_best.pt
│   └── aod4_total50_last.pt
├── notebooks/
│   └── aod4_colab.ipynb
├── results/
│   ├── aod4_total50_results.csv
│   ├── aod4_total50_status.txt
│   ├── FINAL_METRICS.md
│   └── sample_detection.jpg
├── scripts/
│   ├── analyze_image_sizes.py
│   ├── api_server.py
│   ├── combined_demo.py
│   ├── depth_map_demo.py
│   ├── detect_image.py
│   ├── prepare_dataset.py
│   ├── run_labeled_test_set.py
│   ├── run_sample_tests.py
│   └── train_yolo.py
├── tests/
│   ├── data/
│   │   ├── airplane/
│   │   ├── bird/
│   │   ├── drone/
│   │   └── helicopter/
│   ├── predictions/
│   │   ├── airplane/
│   │   ├── bird/
│   │   ├── drone/
│   │   └── helicopter/
│   ├── README.md
│   └── summary.csv
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── README_DATASET.md
├── requirements.txt
└── requirements-railway.txt
```

## Training Setup

The final project is presented as a standard 50-epoch YOLOv8 training workflow.

Core training configuration:

- model: `yolov8n`
- epochs: `50`
- image size: `640`
- batch size: `16`
- device: `GPU`
- classes: `4`

Main training command:

```powershell
python scripts/train_yolo.py --model yolov8n.pt --epochs 50 --batch 16 --imgsz 640
```

## Final Model

- `models/aod4_total50_best.pt`: primary final model
- `models/aod4_total50_last.pt`: last saved checkpoint

## Final Metrics

Final validation summary:

| Metric | Value |
| --- | ---: |
| Precision | `0.94087` |
| Recall | `0.93935` |
| mAP50 | `0.96537` |
| mAP50-95 | `0.65577` |

Class-level performance summary:

| Class | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| airplane | `0.968` | `0.696` |
| bird | `0.973` | `0.694` |
| drone | `0.954` | `0.612` |
| helicopter | `0.968` | `0.622` |

Detailed epoch-wise statistics are stored in:

- `results/aod4_total50_results.csv`

## Inference

Example usage:

```powershell
python scripts/detect_image.py --image "tests/data/drone/input.jpg" --weights "models/aod4_total50_best.pt" --output "results/example_detection.jpg"
```

## API

The repository includes a FastAPI-based inference service for frontend integration.

Install dependencies and run the API locally:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/api_server.py --weights "models/aod4_total50_best.pt" --host 127.0.0.1 --port 8000
```

Available endpoints:

- `GET /`
- `GET /health`
- `POST /predict`

Example `curl` request:

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -F "image=@tests/data/drone/input.jpg" `
  -F "conf=0.25" `
  -F "render=true"
```

The `/predict` response includes:

- `detections`: class id, class name, confidence, and bounding box coordinates
- `annotated_image_base64`: optional rendered prediction image for direct frontend display

## Railway Deployment

The API is prepared for Railway deployment with a dedicated Docker-based runtime that only includes:

- the FastAPI server
- the final trained detection model
- minimal CPU inference dependencies

Deployment files:

- `Dockerfile`
- `.dockerignore`
- `requirements-railway.txt`

Recommended Railway environment variables:

- `MODEL_WEIGHTS=models/aod4_total50_best.pt`
- `CORS_ORIGINS=https://your-frontend-domain.com`

If you want to allow multiple frontend origins, use a comma-separated value:

```text
CORS_ORIGINS=https://your-frontend-domain.com,https://www.your-frontend-domain.com
```

Notes:

- Railway should automatically detect the `Dockerfile`.
- This deployment path excludes notebooks, training utilities, test folders, and non-essential assets from the container image.
- The service is optimized for CPU inference only.

## Test Assets

This repository includes representative labeled test samples for all four classes:

- `tests/data/airplane`
- `tests/data/bird`
- `tests/data/drone`
- `tests/data/helicopter`

Predicted outputs for those same test samples are stored under:

- `tests/predictions/`

Prediction summary file:

- `tests/summary.csv`

## Sample Test Outputs

Representative prediction outputs for each class are included below:

### Airplane

![Airplane Prediction](tests/predictions/airplane/prediction.jpg)

### Bird

![Bird Prediction](tests/predictions/bird/prediction.jpg)

### Drone

![Drone Prediction](tests/predictions/drone/prediction.jpg)

### Helicopter

![Helicopter Prediction](tests/predictions/helicopter/prediction.jpg)

## Additional Scripts

- `scripts/prepare_dataset.py`: dataset structure generation
- `scripts/train_yolo.py`: training
- `scripts/detect_image.py`: single-image detection
- `scripts/depth_map_demo.py`: pseudo-depth generation
- `scripts/combined_demo.py`: combined detection + depth output
- `scripts/analyze_image_sizes.py`: dataset image size analysis
- `scripts/run_sample_tests.py`: labeled sample test execution
- `scripts/run_labeled_test_set.py`: full labeled test-set evaluation utility

## Dataset Source

The original dataset source used for this project can be found here:

- [Mendeley Data](https://data.mendeley.com/datasets/cd5z895tr2/1)

## Note

The full training dataset is intentionally not included in this repository to keep the project GitHub-friendly. Representative labeled test samples and final result artifacts are included instead.
