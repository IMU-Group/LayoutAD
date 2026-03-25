import os
from pathlib import Path
import torch
import json
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans

id2name =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
            'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
            'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
            'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
            'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
            'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']

class LayoutadDataset(Dataset):
    def __init__(self, root:str, split:str='train'):
        self.root = root
        self.split = split
        self.img_list = list(Path(os.path.join(root, split)).rglob('*.jpg')) + \
                        list(Path(os.path.join(root, split)).rglob('*.png'))
        self.approx_eps = 3.0 if split == 'train' else 2.0

        # with open('/home/zengzc/Projects/LayoutAD/src/scene_categories.json', 'r') as f:
        #     scene_data = json.load(f)
        # self.stuff_id = [cat['id'] for cat in scene_data['categories'] if cat['isthing'] == 0]
        # self.id2name = {cat['id']: cat['name'] for cat in scene_data['categories']}

        self.transform = transforms.Compose([
            transforms.ToTensor(),                   
            transforms.Resize((640, 640), antialias=True),        
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        self.clip_features = os.path.join(root, 'clip_features')
        if not os.path.exists(self.clip_features):
            raise FileNotFoundError(f"CLIP feature directory not found: {self.clip_features}")

    def __len__(self):
        return len(self.img_list)
    
    def _load_gt_mask(self, gt_mask_path: str, H:int, W:int) -> np.array:
        if not gt_mask_path or not os.path.exists(gt_mask_path):
            return np.zeros((H, W), dtype=np.uint8)
        gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            return np.zeros((H, W), dtype=np.uint8)
        if gt.shape != (H, W):
            gt = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return (gt > 0).astype(np.uint8)

    @staticmethod
    def _overlap_labels(obj_mask: np.ndarray, gt_mask: np.ndarray) -> int:
        if obj_mask.sum() == 0:
            return 0
        inter = np.logical_and(obj_mask > 0, gt_mask > 0).sum()
        if inter == 0:
            return 0
        obj_area = (obj_mask > 0).sum()
        gt_area = (gt_mask > 0).sum()
        oor = inter / (obj_area + 1e-6)                      # overlap on object
        union = obj_area + gt_area - inter
        iou = inter / (union + 1e-6) if union > 0 else 0.0
        return int((oor >= 0.8) or (iou >= 0.8))

    def __getitem__(self, idx: int):

        img_path = str(self.img_list[idx])
        basename = os.path.basename(img_path)
        # segmentation for image
        label_path = os.path.join(self.root, 'label', basename.replace('.jpg', '.png'))
        # segmentation for object
        json_path = label_path.replace('.png', '.json')
        # mask path for anomaly image
        gt_mask_path = ''
        if self.split == 'test':
            gt_mask_path = img_path.replace('/test/', '/ground_truth/')
            gt_mask_path = gt_mask_path if os.path.exists(gt_mask_path) else ''

        panoptic_seg = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)
        with open(json_path, 'r') as f:
            img_json = json.load(f)
        segments_info = img_json['segments']

        # clip
        clip_feat_path = os.path.join(self.clip_features, basename.replace('.jpg', '.pt').replace('.png', '.pt'))
        if not os.path.exists(clip_feat_path):
            raise FileNotFoundError(f"Missing CLIP feature map for {basename}")
        feat_map = torch.load(clip_feat_path)
        C, Hf, Wf = feat_map.shape

        H, W = panoptic_seg.shape
        mask_small = cv2.resize(panoptic_seg, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
        
        nodes, node_labels = [], []
        gt_mask_bin = self._load_gt_mask(gt_mask_path, H, W)

        for segment in segments_info:
            seg_id = segment['id']
            cat_id = segment['category_id']
            # is_thing = int(int(cat_id) not in self.stuff_id)
            is_thing = segment['isthing']
            area = segment['area']

            mask = (panoptic_seg == seg_id).astype(np.uint8)
            if mask.sum() < 500:
                continue 
            
            # geometric feature
            x, y, w, h = cv2.boundingRect(mask)
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + w / 2, y + h / 2
            # 归一化
            cx /= W; cy /= H; w /= W; h /= H; area /= (W * H)
            aspect_ratio = w / (h + 1e-6)
            size_ratio = area / (h * w + 1e-6)
            hu = cv2.HuMoments(M).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

            # visual feature
            mask_down = (mask_small == seg_id).astype(np.float32)
            mask_t = torch.tensor(mask_down).unsqueeze(0)
            if mask_t.sum() > 1:
                feat = (feat_map * mask_t).sum(dim=(1, 2)) / (mask_t.sum() + 1e-5)
                feat = F.normalize(feat, dim=0)
            else:
                feat = torch.zeros(C)
            
            # node label
            if (self.split == 'test') and (gt_mask_bin is not None):
                mask_bin = (mask > 0).astype(np.uint8)
                node_label = self._overlap_labels(
                    mask_bin, gt_mask_bin)
            else:
                node_label = 0

            node_feature = {
                "id": int(seg_id),
                "category_id": int(cat_id),
                "category_name": id2name[cat_id],
                "is_thing": is_thing,
                "x": x / W,
                "y": y / H,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "size_ratio": size_ratio,
                "hu_moments": hu.tolist(),
                "vis_feat": feat,
                'text_feature': segment.get('token_feature', ''),
                "mask": mask,
                "node_label": int(node_label),
            }

            nodes.append(node_feature)
            node_labels.append(int(node_label))

        sample = {
            "image": img_tensor,
            "image_path": img_path,
            "gt_mask_path": gt_mask_path,
            "image_size": (H, W),
            "nodes": nodes,
            'label': int("good" not in img_path),
            "node_labels": np.array(node_labels, dtype=np.int32)
        }
        return sample
