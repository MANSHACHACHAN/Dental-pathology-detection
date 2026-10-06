import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.model_selection import train_test_split
from train import TuftsDetectionDataset, parse_bbox_json

CHECKPOINT   = '/content/best_fasterrcnn_tufts.pth'
RAD_DIR      = '/content/tufts/Radiographs/Radiographs'
BBOX_JSON    = '/content/tufts/Segmentation/Segmentation/teeth_bbox.json'
NUM_CLASSES  = 53
SCORE_THRESH = 0.5


def load_model(ckpt_path, num_classes):
    # same architecture as training, just loading saved weights
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    ckpt  = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.cuda().eval()
    print(f"loaded epoch {ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")
    return model, ckpt['label_map'], ckpt['idx_to_tooth']


def evaluate_map(model, val_loader):
    metric = MeanAveragePrecision(iou_type="bbox")
    model.eval()

    with torch.no_grad():
        for images, targets in val_loader:
            images  = [img.cuda() for img in images]
            outputs = model(images)
            preds = [{'boxes':  o['boxes'].cpu(),
                      'scores': o['scores'].cpu(),
                      'labels': o['labels'].cpu()} for o in outputs]
            gts   = [{'boxes':  t['boxes'].cpu(),
                      'labels': t['labels'].cpu()} for t in targets]
            metric.update(preds, gts)

    r = metric.compute()
    print(f"\nmAP@0.50      : {r['map_50']:.4f}")
    print(f"mAP@0.75      : {r['map_75']:.4f}")
    print(f"mAP@0.50:0.95 : {r['map']:.4f}")


def visualize_predictions(model, val_ds, idx_to_tooth, n=3):
    # pick random val images, draw ground truth (green) vs predictions (red)
    model.eval()
    idxs = np.random.choice(len(val_ds), n, replace=False)
    fig, axes = plt.subplots(n, 1, figsize=(16, 7*n))
    if n == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, idx in zip(axes, idxs):
            img_t, target = val_ds[idx]
            out = model([img_t.cuda()])[0]

            ax.imshow(img_t.permute(1, 2, 0).numpy(), cmap='gray')

            # green dashed = ground truth boxes
            for box in target['boxes']:
                x1, y1, x2, y2 = box.tolist()
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    lw=1.5, edgecolor='lime',
                    facecolor='none', linestyle='--'))

            # red solid = predictions above threshold
            for box, score, lbl in zip(out['boxes'],
                                        out['scores'],
                                        out['labels']):
                if score < SCORE_THRESH:
                    continue
                x1, y1, x2, y2 = box.cpu().tolist()
                tooth = idx_to_tooth.get(lbl.item(), '?')
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    lw=1.5, edgecolor='red', facecolor='none'))
                ax.text(x1, y1-4, f"{tooth}:{score:.2f}",
                        color='red', fontsize=7,
                        bbox=dict(facecolor='white', alpha=0.4, pad=1))

            ax.legend(handles=[
                Line2D([0],[0], color='lime', lw=2,
                       linestyle='--', label='ground truth'),
                Line2D([0],[0], color='red',  lw=2,
                       label=f'predicted (>= {SCORE_THRESH})'),
            ], loc='upper right')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('/content/fasterrcnn_predictions.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    df_bbox, _, _ = parse_bbox_json(BBOX_JSON)
    all_ids    = df_bbox['image_id'].unique().tolist()
    _, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

    val_ds     = TuftsDetectionDataset(val_ids, df_bbox, RAD_DIR)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=2,
                            collate_fn=lambda b: tuple(zip(*b)))

    model, label_map, idx_to_tooth = load_model(CHECKPOINT, NUM_CLASSES)
    evaluate_map(model, val_loader)
    visualize_predictions(model, val_ds, idx_to_tooth)
