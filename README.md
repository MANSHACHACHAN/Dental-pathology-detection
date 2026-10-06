# Dental Pathology Detection — Faster R-CNN vs YOLOv8

Comparing two object detection models on the Tufts Dental Database 2022
for tooth localization and dental lesion detection in X-rays.

---

## Results

**Faster R-CNN (ResNet50-FPN) — Tooth Detection**
- 52 classes (Universal Numbering System)
- mAP@0.5 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : 0.5540
- mAP@0.5:0.95 : 0.3236
- Val Loss &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : 0.9837

**YOLOv8m — Caries Detection**
- 1 class (caries)
- mAP@0.5 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : 0.0723
- mAP@0.5:0.95 : 0.0429
- Precision &nbsp;&nbsp;&nbsp;&nbsp; : 0.2647
- Recall &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : 0.1667

Faster R-CNN performed significantly better on the structured
tooth localization task. YOLOv8m was challenged by the small
size of caries lesions (avg ~5% of image width) and limited
annotated samples per class.

---

## Dataset

[Tufts Dental Database 2022](https://www.kaggle.com/datasets/tommyngx/the-tufts-dental-database-2022)

- 1000 panoramic dental X-rays
- 26,005 bounding box annotations
- 52 tooth classes — Universal Numbering System (teeth 1–32 + primary teeth A–T)
- Expert binary masks for caries/lesion regions
- CLAHE preprocessing applied to all images

---

## Repository Structure

```
dental-pathology-detection/
├── faster_rcnn/
│   ├── train.py            # annotation parsing, dataset class, training loop
│   └── evaluate.py         # mAP evaluation + prediction visualization
├── yolov8/
│   └── train_evaluate.py   # dataset prep, training, MC-dropout inference,
│                           # uncertainty estimation, clinical report
├── visualize_unseen.py     # run both models on unseen images
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/MANSHACHACHAN/Dental-pathology-detection
cd dental-pathology-detection
pip install -r requirements.txt
```

Download dataset:
```bash
kaggle datasets download -d tommyngx/the-tufts-dental-database-2022
unzip the-tufts-dental-database-2022.zip -d tufts/
```

---

## Usage

**Train Faster R-CNN:**
```bash
cd faster_rcnn
python train.py
```

**Evaluate Faster R-CNN:**
```bash
python evaluate.py
```

**Train + Evaluate YOLOv8:**
```bash
cd yolov8
python train_evaluate.py
```

**Visualize on unseen images:**
```bash
python visualize_unseen.py
```

---

## Model Details

### Faster R-CNN
- Backbone: ResNet50 + FPN pretrained on ImageNet
- 52 tooth classes using Universal Numbering System
- Input resized to 800px long side with CLAHE enhancement
- SGD optimizer, StepLR scheduler (lr=0.005, step=7, gamma=0.1)
- 20 epochs, batch size 4, 80/20 train/val split

### YOLOv8m
- Pretrained YOLOv8m fine-tuned for caries detection
- Binary masks from expert annotations converted to YOLO format
- MC-Dropout uncertainty estimation with weighted boxes fusion
- Clinical severity staging: D0 suspect → D3 advanced
- AdamW optimizer, cosine LR, 60 epochs with early stopping (patience=20)

---

## Key Findings

- Faster R-CNN achieved strong tooth localization (mAP@0.5: 0.554)
  on a challenging 52-class problem
- YOLOv8m struggled with very small lesion boxes
  (avg ~5% of image width) and limited training samples
- MC-Dropout uncertainty flagging helped identify low-confidence
  predictions for clinical referral

