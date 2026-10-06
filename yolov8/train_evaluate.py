import os, cv2, glob, shutil, csv
import numpy as np
import torch.nn as nn
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from datetime import datetime

RAD_DIR       = '/content/tufts/Radiographs/Radiographs'
MASK_DIR      = '/content/tufts/Expert/Expert/mask'
EXP_JSON      = '/content/tufts/Expert/Expert/expert.json'
DATASET       = '/content/dental_yolo'
MC_PASSES     = 20
CONF_THRESH   = 0.15
IOU_THRESH    = 0.40
UQ_THRESHOLD  = 0.12   # sigma above this = uncertain prediction
DET_THRESHOLD = 0.70   # detection rate below this = also uncertain


def apply_clahe(img_bgr):
    # boosts local contrast — helps with subtle lesion boundaries
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def mask_to_yolo_bboxes(mask_path, min_area=100):
    # read binary mask, find white blobs, return YOLO-format xywh (normalised)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], (0, 0)
    H, W = mask.shape
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    bboxes = []
    for i in range(1, num_labels):
        # skip tiny blobs — usually scan artifacts, not real lesions
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT];  y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append((
            max(0., min(1., (x + w/2) / W)),
            max(0., min(1., (y + h/2) / H)),
            max(0., min(1., w / W)),
            max(0., min(1., h / H))
        ))
    return bboxes, (H, W)


def build_dataset():
    # match each radiograph to its expert mask, skip pairs with empty masks
    mask_lookup = {Path(f).stem.lower(): f'{MASK_DIR}/{f}'
                   for f in os.listdir(MASK_DIR)}
    img_files   = [f for f in os.listdir(RAD_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_lookup  = {Path(f).stem.lower(): f for f in img_files}

    valid_pairs = []
    for stem, mask_path in mask_lookup.items():
        bboxes, _ = mask_to_yolo_bboxes(mask_path)
        if not bboxes or stem not in img_lookup:
            continue
        valid_pairs.append((img_lookup[stem], mask_path, bboxes))

    print(f"images with lesion regions: {len(valid_pairs)}")

    train_p, temp   = train_test_split(valid_pairs, test_size=0.3,  random_state=42)
    val_p,   test_p = train_test_split(temp,        test_size=0.33, random_state=42)
    print(f"train:{len(train_p)} val:{len(val_p)} test:{len(test_p)}")

    for split in ['train', 'val', 'test']:
        shutil.rmtree(f'{DATASET}/images/{split}', ignore_errors=True)
        shutil.rmtree(f'{DATASET}/labels/{split}', ignore_errors=True)
        os.makedirs(f'{DATASET}/images/{split}', exist_ok=True)
        os.makedirs(f'{DATASET}/labels/{split}', exist_ok=True)

    def write_split(pairs, split):
        img_out   = f'{DATASET}/images/{split}'
        label_out = f'{DATASET}/labels/{split}'
        for img_file, mask_path, bboxes in pairs:
            img = cv2.imread(f'{RAD_DIR}/{img_file}')
            if img is None:
                continue
            H, W = img.shape[:2]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # if mask resolution differs from image, resize and recompute boxes
            if mask is not None and mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite('/tmp/_resized.png', mask)
                bboxes, _ = mask_to_yolo_bboxes('/tmp/_resized.png')
            cv2.imwrite(f'{img_out}/{img_file}', apply_clahe(img))
            with open(f'{label_out}/{Path(img_file).stem}.txt', 'w') as lf:
                for xc, yc, bw, bh in bboxes:
                    # skip boxes < 0.5% image size — likely noise
                    if bw > 0.005 and bh > 0.005:
                        lf.write(f'0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')

    write_split(train_p, 'train')
    write_split(val_p,   'val')
    write_split(test_p,  'test')

    yaml = (f"path: {DATASET}\ntrain: images/train\n"
            f"val: images/val\ntest: images/test\nnc: 1\nnames:\n  0: caries\n")
    with open(f'{DATASET}/data.yaml', 'w') as f:
        f.write(yaml)
    print("dataset written")


def train_yolo():
    model = YOLO('yolov8m.pt')
    model.train(
        data=f'{DATASET}/data.yaml',
        epochs=60,
        imgsz=640,
        batch=8,
        lr0=0.001,
        cos_lr=True,
        patience=20,       # stop early if val stalls for 20 epochs
        dropout=0.1,
        project='/content/runs',
        name='caries',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        verbose=True
    )
    print("training done")


def evaluate_yolo():
    best = YOLO('/content/runs/caries/weights/best.pt')
    metrics = best.val(
        data=f'{DATASET}/data.yaml',
        split='test', conf=0.25, iou=0.45
    )
    print(f"mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"precision   : {metrics.box.p.mean():.4f}")
    print(f"recall      : {metrics.box.r.mean():.4f}")


def enable_dropout(yolo_model):
    # YOLOv8 has no dropout by default — this activates it if present
    found = False
    for m in yolo_model.model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            found = True
    return found


def mc_predict(yolo_model, img_path, n=MC_PASSES):
    # run N forward passes; if no dropout exists, vary input noise instead
    has_dropout = enable_dropout(yolo_model)
    img_bgr = cv2.imread(img_path)
    H, W    = img_bgr.shape[:2]
    all_boxes, all_scores = [], []

    for i in range(n):
        if has_dropout:
            res = yolo_model.predict(img_path, conf=CONF_THRESH,
                                     iou=IOU_THRESH, verbose=False)[0]
        else:
            # vary noise level across passes to simulate input uncertainty
            noise_std = 5.0 * (i % 4 + 1)
            noisy = np.clip(
                img_bgr.astype(np.float32) +
                np.random.normal(0, noise_std, img_bgr.shape),
                0, 255).astype(np.uint8)
            cv2.imwrite('/tmp/_mc_tmp.jpg', noisy)
            res = yolo_model.predict('/tmp/_mc_tmp.jpg', conf=CONF_THRESH,
                                     iou=IOU_THRESH, verbose=False)[0]

        if len(res.boxes):
            all_boxes.append(
                np.clip(res.boxes.xyxy.cpu().numpy() /
                        [W, H, W, H], 0, 1).tolist())
            all_scores.append(res.boxes.conf.cpu().numpy().tolist())
        else:
            all_boxes.append([]); all_scores.append([])

    return all_boxes, all_scores, H, W


def estimate_uncertainty(all_scores):
    if not all_scores:
        return 0., 0., 0.
    max_per_pass   = [max(s) if s else 0. for s in all_scores]
    detection_rate = sum(1 for s in all_scores if s) / len(all_scores)
    mu    = float(np.mean(max_per_pass))
    sigma = float(np.std(max_per_pass))
    # if model only detects in some passes, bump up sigma to reflect that
    sigma_adj = sigma + (1 - detection_rate) * 0.15
    return mu, sigma_adj, detection_rate


def wbf_predict(img_path, yolo_model):
    # fuse boxes from all MC passes using weighted boxes fusion
    all_boxes, all_scores, H, W = mc_predict(yolo_model, img_path)
    mu, sigma, det_rate = estimate_uncertainty(all_scores)

    if not any(all_boxes):
        return np.array([]), np.array([]), mu, sigma, det_rate, H, W

    valid_b = [b for b in all_boxes  if b]
    valid_s = [s for s in all_scores if s]
    labels  = [[0]*len(s) for s in valid_s]

    fused_b, fused_s, _ = weighted_boxes_fusion(
        valid_b, valid_s, labels,
        iou_thr=IOU_THRESH, skip_box_thr=CONF_THRESH)

    fused_b_px = fused_b * np.array([W, H, W, H]) if len(fused_b) else fused_b
    return fused_b_px, fused_s, mu, sigma, det_rate, H, W


def severity_from_conf(conf):
    # rough clinical staging based on detection confidence
    if conf >= 0.75: return 'D3-advanced',  (220,  30,  30)
    if conf >= 0.50: return 'D2-moderate',  (220, 120,   0)
    if conf >= 0.25: return 'D1-early',     ( 30, 160,  30)
    return                   'D0-suspect',  (100, 100, 200)


def visualize_and_log(img_path, yolo_model, results_log):
    fused_b, fused_s, mu, sigma, det_rate, H, W = wbf_predict(img_path, yolo_model)
    img = np.array(Image.open(img_path).convert('RGB'))

    fig, axes = plt.subplots(1, 2, figsize=(24, 7))
    for ax in axes:
        ax.imshow(img, cmap='gray'); ax.axis('off')
    axes[0].set_title('Enhanced X-ray', fontweight='bold', fontsize=12)

    uq_flag = (sigma > UQ_THRESHOLD) or \
              (det_rate < DET_THRESHOLD and len(fused_b) > 0)
    status  = 'UNCERTAIN — refer clinician' if uq_flag else 'confident'
    axes[1].set_title(
        f"{len(fused_b)} lesions | mu={mu:.2f} sigma={sigma:.3f} "
        f"det={det_rate*100:.0f}% | {status}",
        fontweight='bold', fontsize=9,
        color='crimson' if uq_flag else 'darkgreen')

    for box, score in zip(fused_b, fused_s):
        x1, y1, x2, y2 = box
        sev, rgb = severity_from_conf(float(score))
        cm = [c/255 for c in rgb]
        axes[1].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            lw=0, facecolor=cm, alpha=0.20))
        axes[1].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            lw=2.5, edgecolor=cm, facecolor='none'))
        axes[1].text(x1+2, y1-8, f"{sev} {score:.2f}",
                     color='white', fontsize=8, fontweight='bold',
                     bbox=dict(facecolor=cm, alpha=0.9,
                               pad=2, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(f'/content/caries_{Path(img_path).stem}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    counts = {'D3': 0, 'D2': 0, 'D1': 0, 'D0': 0}
    for s in fused_s:
        lbl, _ = severity_from_conf(float(s))
        counts[lbl[:2]] += 1

    results_log[Path(img_path).name] = {
        'n_lesions': len(fused_b), 'mean_conf': round(float(mu), 4),
        'sigma':     round(float(sigma), 4),
        'det_rate':  round(float(det_rate), 3),
        'uq_flag':   uq_flag, 'severity': counts
    }


def save_report(results_log, out_path='/content/cariescope_report.csv'):
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'n_lesions', 'mean_conf', 'sigma',
                         'detection_rate', 'uq_flag',
                         'D0', 'D1', 'D2', 'D3',
                         'clinical_action', 'timestamp'])
        for name, info in results_log.items():
            sv = info['severity']
            if info['uq_flag']:
                action = 'refer to clinician'
            elif sv.get('D3', 0) > 0:
                action = 'urgent treatment'
            elif sv.get('D2', 0) > 0:
                action = 'treatment recommended'
            elif sv.get('D1', 0) > 0:
                action = 'monitor + preventive care'
            else:
                action = 'no caries detected'
            writer.writerow([
                name, info['n_lesions'], info['mean_conf'],
                info['sigma'], info['det_rate'],
                'YES' if info['uq_flag'] else 'NO',
                sv.get('D0',0), sv.get('D1',0),
                sv.get('D2',0), sv.get('D3',0),
                action, datetime.now().strftime('%Y-%m-%d %H:%M')
            ])
    print(f"report saved to {out_path}")


if __name__ == '__main__':
    build_dataset()
    train_yolo()
    evaluate_yolo()

    best = YOLO('/content/runs/caries/weights/best.pt')
    test_imgs = (glob.glob(f'{DATASET}/images/test/*.JPG') +
                 glob.glob(f'{DATASET}/images/test/*.jpg'))[:8]

    results_log = {}
    for p in test_imgs:
        visualize_and_log(p, best, results_log)

    save_report(results_log)
