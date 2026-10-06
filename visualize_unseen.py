import os, cv2, glob, torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

# since checkpoints are on drive, load from there
DRIVE_DIR    = '/content/drive/MyDrive/dental_project'
RAD_DIR      = '/content/tufts/Radiographs/Radiographs'
RCNN_CKPT    = f'{DRIVE_DIR}/best_fasterrcnn_tufts.pth'
YOLO_CKPT    = f'{DRIVE_DIR}/yolo_runs/caries/weights/best.pt'
SCORE_THRESH = 0.5
CONF_THRESH  = 0.15

# ── load faster r-cnn ─────────────────────────────────────────────────────────
import sys
sys.path.append(f'{DRIVE_DIR}/faster_rcnn')
from train import get_model

def load_rcnn(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location='cuda')
    model = get_model(53)
    model.load_state_dict(ckpt['model_state_dict'])
    model.cuda().eval()
    print(f"rcnn loaded — epoch {ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")
    return model, ckpt['idx_to_tooth']

# ── faster r-cnn inference on one image ───────────────────────────────────────
def rcnn_predict(model, img_path, idx_to_tooth, score_thresh=SCORE_THRESH):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w  = img.shape[:2]
    scale = 800 / float(max(h, w))
    img_r = cv2.resize(img, (int(w*scale), int(h*scale)))
    tensor = torch.from_numpy(
        img_r.astype(np.float32) / 255.
    ).permute(2, 0, 1).unsqueeze(0).cuda()

    with torch.no_grad():
        out = model(tensor)[0]

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.imshow(img_r, cmap='gray')

    count = 0
    for box, score, lbl in zip(out['boxes'], out['scores'], out['labels']):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.cpu().tolist()
        tooth = idx_to_tooth.get(lbl.item(), '?')
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            lw=1.5, edgecolor='red', facecolor='none'))
        ax.text(x1, y1-4, f"{tooth}:{score:.2f}",
                color='red', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.4, pad=1))
        count += 1

    ax.set_title(f"Faster R-CNN | {Path(img_path).name} | "
                 f"{count} teeth detected (score>={score_thresh})",
                 fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
