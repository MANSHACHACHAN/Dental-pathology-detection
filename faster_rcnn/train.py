import os, json, time, cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.model_selection import train_test_split

RAD_DIR     = '/content/tufts/Radiographs/Radiographs'
BBOX_JSON   = '/content/tufts/Segmentation/Segmentation/teeth_bbox.json'
NUM_CLASSES = 53   # 52 teeth + background
BATCH_SIZE  = 4
NUM_EPOCHS  = 20
IMG_SIZE    = 800
SAVE_DIR    = '/content'


def parse_bbox_json(json_path):
    with open(json_path) as f:
        bbox_data = json.load(f)

    records = []
    for entry in bbox_data:
        # external ID is a full path like ".../797.JPG", we just want "797"
        img_id = os.path.splitext(os.path.basename(entry['External ID']))[0]
        for obj in entry['Label']['objects']:
            bbox = obj['bounding box']
            records.append({
                'image_id': img_id,
                'tooth_id': obj['title'],   # e.g. "1", "A", "32"
                'x': bbox[0], 'y': bbox[1],
                'w': bbox[2], 'h': bbox[3]
            })

    df = pd.DataFrame(records)

    # tooth IDs are strings like "1","2"..."32","A","B"..."T"
    # sort numerics first then letters, map each to an int
    # 0 is reserved for background in Faster R-CNN
    all_ids = sorted(df['tooth_id'].unique(),
                     key=lambda x: (x.isalpha(), int(x) if x.isdigit() else x))
    label_map    = {tid: i+1 for i, tid in enumerate(all_ids)}
    idx_to_tooth = {v: k for k, v in label_map.items()}
    df['label']  = df['tooth_id'].map(label_map)

    print(f"unique tooth classes : {df['tooth_id'].nunique()}")
    print(f"total bounding boxes : {len(df)}")
    return df, label_map, idx_to_tooth


class TuftsDetectionDataset(Dataset):
    def __init__(self, image_ids, df_bbox, img_dir, img_size=800):
        self.image_ids = image_ids
        self.df        = df_bbox
        self.img_dir   = img_dir
        self.img_size  = img_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # files on disk are uppercase .JPG but annotations have lowercase
        # try both just to be safe
        img_path = None
        for ext in ['.JPG', '.jpg']:
            c = os.path.join(self.img_dir, f"{img_id}{ext}")
            if os.path.exists(c):
                img_path = c
                break
        if img_path is None:
            raise FileNotFoundError(f"no image found for id: {img_id}")

        img   = cv2.imread(img_path)
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize so the longer side = img_size, keep aspect ratio
        h, w  = img.shape[:2]
        scale = self.img_size / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img   = img.astype(np.float32) / 255.0

        rows = self.df[self.df['image_id'] == img_id]
        boxes, labels = [], []
        for _, row in rows.iterrows():
            # scale box coords to match the resized image
            x1 = max(0., min(float(row['x']) * scale, new_w))
            y1 = max(0., min(float(row['y']) * scale, new_h))
            x2 = max(0., min((float(row['x'])+float(row['w'])) * scale, new_w))
            y2 = max(0., min((float(row['y'])+float(row['h'])) * scale, new_h))
            # skip degenerate boxes that can appear at image edges after scaling
            if x2 > x1+1 and y2 > y1+1:
                boxes.append([x1, y1, x2, y2])
                labels.append(int(row['label']))

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes':    boxes,
            'labels':   labels,
            'image_id': torch.tensor([idx])
        }
        return torch.from_numpy(img).permute(2, 0, 1), target


def get_model(num_classes):
    # start from pretrained weights, only swap out the final head
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(df_bbox, img_dir, num_classes, save_dir, label_map, idx_to_tooth):
    all_ids = df_bbox['image_id'].unique().tolist()
    train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    print(f"train: {len(train_ids)} | val: {len(val_ids)}")

    collate_fn   = lambda b: tuple(zip(*b))
    train_loader = DataLoader(
        TuftsDetectionDataset(train_ids, df_bbox, img_dir),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        TuftsDetectionDataset(val_ids, df_bbox, img_dir),
        batch_size=2, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    model     = get_model(num_classes).cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.005, momentum=0.9, weight_decay=0.0005)
    # lr drops by 10x at epoch 7 and again at epoch 14
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_val  = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for i, (images, targets) in enumerate(train_loader):
            images  = [img.cuda() for img in images]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            # clip gradients — helps when a bad batch causes a spike
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += losses.item()

            if i % 20 == 0:
                print(f"  ep{epoch+1} b{i}/{len(train_loader)} "
                      f"loss:{losses.item():.4f} "
                      f"cls:{loss_dict['loss_classifier'].item():.3f} "
                      f"box:{loss_dict['loss_box_reg'].item():.3f}")

        # val loss needs train mode to get the loss dict back from the model
        model.train()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images  = [img.cuda() for img in images]
                targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
                val_loss += sum(model(images, targets).values()).item()

        avg_t = epoch_loss / len(train_loader)
        avg_v = val_loss   / len(val_loader)
        scheduler.step()
        print(f"\nepoch {epoch+1}/{NUM_EPOCHS} "
              f"train:{avg_t:.4f} val:{avg_v:.4f} "
              f"time:{time.time()-t0:.1f}s")

        if avg_v < best_val:
            best_val = avg_v
            torch.save({
                'epoch':            epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss':         avg_v,
                'label_map':        label_map,
                'idx_to_tooth':     idx_to_tooth,
            }, f'{save_dir}/best_fasterrcnn_tufts.pth')
            print(f"  saved best (val={avg_v:.4f})")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       f'{save_dir}/fasterrcnn_ep{epoch+1}.pth')

    print(f"\ndone. best val loss: {best_val:.4f}")


if __name__ == '__main__':
    df_bbox, label_map, idx_to_tooth = parse_bbox_json(BBOX_JSON)
    train(df_bbox, RAD_DIR, NUM_CLASSES, SAVE_DIR, label_map, idx_to_tooth)
