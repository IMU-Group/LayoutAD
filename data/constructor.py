import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.data import Data, Batch
from typing import List, Dict, Any, Tuple
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans
from scipy.ndimage import convolve

from torch.utils.data import DataLoader

def skeleton_stats(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 10:
        return 0.0, 0.0
    sk = skeletonize(m > 0)
    sk_len = float(sk.sum())
    kernel = np.ones((3, 3), dtype=int)
    neigh = convolve(sk.astype(int), kernel, mode='constant', cval=0) - sk.astype(int)
    branch_pts = float(((neigh >= 3) & sk).sum())
    sk_len_norm = sk_len / (float(m.sum()) + 1e-5)
    return sk_len_norm, branch_pts

def mask_orientation(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0 and len(cnts[0]) >= 5:
        try:
            (_, _), (_MA, _ma), angle = cv2.fitEllipse(cnts[0])  # 0~180
            return float(angle / 180.0)
        except cv2.error:
            pass
    ys, xs = np.where(m > 0)
    if len(xs) >= 3:
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        pts -= pts.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        v = vt[0]
        ang = np.arctan2(v[1], v[0])
        ang = (ang + np.pi) % np.pi 
        return float(ang / np.pi)
    return 0.0

class GraphConstructor(nn.Module):
    def __init__(self,
                 graph_type: str = 'knn_threshold',
                 k: int = 8,
                 distance_threshold: float = 0.25,
                 ):
        assert graph_type in ('knn', 'threshold', 'knn_threshold')
        self.graph_type = graph_type
        self.k = k
        self.distance_threshold = distance_threshold

    def _geo_feature(self, node: Dict[str, Any]):

        base = torch.tensor([
            node['cx'], node['cy'], node['w'], node['h'],
            node['area'], node['aspect_ratio'], node['size_ratio']
        ], dtype=torch.float32) # 7
        base[:5] = base[:5].clamp(0, 1)

        hu = torch.tensor(node['hu_moments'], dtype=torch.float32) # 7
        hu = (hu - hu.mean()) / (hu.std() + 1e-6)

        sk_len_norm, branch_pts = skeleton_stats(node['mask']) # 1 + 1
        branch_pts = np.log1p(branch_pts)
        orientation = mask_orientation(node['mask']) # 1

        extra = torch.tensor([sk_len_norm, branch_pts, orientation], dtype=torch.float32)
        extra = (extra - extra.mean()) / (extra.std() + 1e-6)

        geo = torch.cat([base, hu, extra], dim=0)
        geo = (geo - geo.mean()) / (geo.std() + 1e-6)
        return geo   
    
    @staticmethod
    def _calc_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return float(intersection) / max(float(union), 1e-5)
    
    def _extract_geo_edge(self, node1: Dict[str, Any], node2: Dict[str, Any]):
        delta_x = node2["cx"] - node1["cx"]
        delta_y = node2["cy"] - node1["cy"]
        dist = np.sqrt(delta_x**2 + delta_y**2)

        angle = np.arctan2(delta_y, delta_x)
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)

        w_ratio = min(node1["w"], node2["w"]) / max(node1["w"], node2["w"])
        h_ratio = min(node1["h"], node2["h"]) / max(node1["h"], node2["h"])
        area_ratio = min(node1["area"], node2["area"]) / max(node1["area"], node2["area"])

        iou = self._calc_iou(node1['mask'], node2['mask'])

        edge_feat = torch.tensor([
            delta_x, delta_y, dist,
            sin_angle, cos_angle,
            w_ratio, h_ratio, area_ratio, iou
        ], dtype=torch.float32)

        return edge_feat, dist
    
    @staticmethod
    def _vis_text_concat(vis_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        vt = torch.cat([vis_feat, text_feat], dim=-1)  # [D_vis + D_text]
        return F.normalize(vt, dim=-1)
    
    @staticmethod
    def _pairwise_cos(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.numel() == 0 or b.numel() == 0:
            return 0.0
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()

    def _select_neighbors(self, nodes: List[Dict[str, Any]], i: int) -> List[int]:
        dists = []
        for j in range(len(nodes)):
            if j == i:
                continue
            if nodes[i]['is_thing'] == nodes[j]['is_thing'] and nodes[i]['is_thing'] == 0:
                continue
            dx = nodes[j]["cx"] - nodes[i]["cx"]
            dy = nodes[j]["cy"] - nodes[i]["cy"]
            d = float(np.sqrt(dx*dx + dy*dy))
            dists.append((j, d))
        if not dists:
            return []
        dists.sort(key=lambda x: x[1])
        if self.graph_type == 'knn':
            k = min(self.k, len(dists))
            return [j for (j, _) in dists[:k]]
        elif self.graph_type == 'threshold':
            neigh = [j for (j, d) in dists if d < self.distance_threshold]
            if not neigh:
                neigh = [dists[0][0]]
            return neigh
        else:  #  'knn_threshold'
            neigh = [j for (j, d) in dists if d < self.distance_threshold]
            if not neigh:
                neigh = [dists[0][0]]
            if len(neigh) < self.k:
                for (j, _) in dists:
                    if j not in neigh:
                        neigh.append(j)
                        if len(neigh) >= self.k:
                            break
            return neigh

    def __call__(self, batch: List[Dict[str, Any]]) -> Batch:
        graphs = []

        for item in batch:
            nodes = item["nodes"]
            num_nodes = len(nodes)
            if num_nodes == 0:
                raise ValueError(f"No nodes found in {item['image_path']}")
            
            # --- 节点特征 ---
            x_geo = torch.stack([self._geo_feature(n) for n in nodes], dim=0)
            # x_geo = (x_geo - x_geo.mean(dim=0, keepdim=True)) / (x_geo.std(dim=0, keepdim=True) + 1e-6)

            x_vis_only = torch.stack([n['vis_feat'] for n in nodes], dim = 0)
            x_vis_only = F.normalize(x_vis_only, dim = 1)
            # print(x_vis_only.shape)
            x_text = torch.stack([torch.tensor(n['text_feature']) for n in nodes], dim = 0)
            # print(x_text.shape)
            x_vis = F.normalize(torch.cat([x_vis_only, x_text], dim=-1), dim=-1)

            edge_index, edge_geo_attr = [], []
            for i in range(num_nodes):
                neigh = self._select_neighbors(nodes, i)
                for j in neigh:
                    e_ge, _d = self._extract_geo_edge(nodes[i], nodes[j])
                    edge_index.append( (i, j) )
                    edge_geo_attr.append(e_ge)

            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
                edge_attr_geo = torch.stack(edge_geo_attr, dim=0)  
                edge_attr_geo = (edge_attr_geo - edge_attr_geo.mean(dim=0, keepdim=True)) / \
                    (edge_attr_geo.std(dim=0, keepdim=True) + 1e-6)
                # print(edge_attr_geo.mean(), edge_attr_geo.std())                       # [E, 9]
                src, dst = edge_index[0], edge_index[1]
                vis_sims = F.cosine_similarity(x_vis_only[src], x_vis_only[dst], dim=-1).unsqueeze(-1)
                txt_sims = F.cosine_similarity(x_text[src], x_text[dst], dim=-1).unsqueeze(-1)
                edge_attr_vis = torch.cat([vis_sims, txt_sims], dim=-1)  # [E, 2]
                # print(edge_attr_vis.mean(), edge_attr_vis.std())
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr_geo = torch.zeros((0, 9), dtype=torch.float32)
                edge_attr_vis = torch.zeros((0, 2), dtype=torch.float32)
            
            node_labels = torch.from_numpy(item['node_labels']).long()
            is_thing_list = torch.stack([torch.tensor(n['is_thing']) for n in nodes], dim=0)
            
            graphs.append(Data(
                x_geo=x_geo,
                x_vis=x_vis,
                edge_index=edge_index,
                edge_attr_geo=edge_attr_geo,
                edge_attr_vis=edge_attr_vis,
                num_nodes=num_nodes,
                image_path=item['image_path'],
                gt_mask=item['gt_mask_path'],
                img=item['image'],
                label=item['label'],
                node_mask=[n['mask'] for n in nodes],
                image_size=item['image_size'],
                node_labels=node_labels,
                is_thing=is_thing_list
            ))

        return Batch.from_data_list(graphs)
