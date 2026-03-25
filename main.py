import os
import yaml
import torch
import cv2
import time
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import LayoutadDataset
from data.constructor import GraphConstructor
from models.layoutad import LayoutAD
from models.loss import unsup_loss
from utils.metric import evaluate_metrics
from utils.draw import plot_metrics
from utils.tools import build_pixel_score_map
from utils.aggregate import score_aggregate

def ema_update(buf, val, m=0.9):
    buf.copy_(m * buf + (1 - m) * val)

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # load dataset
    print("Loading data...")
    train_dataset = LayoutadDataset(args.data_root, split='train')
    val_dataset = LayoutadDataset(args.data_root, split='test')
    graph_constructor = GraphConstructor()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=graph_constructor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_constructor,
    )
    print("Data loaded successfully.")

    # load model
    model = LayoutAD(
        d_model=args.d_model,
        num_enc_layers=args.num_enc_layers, 
        num_gnn_layers=args.num_gnn_layers
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{args.save_dir}/{now}"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics.csv")

    # train and test
    best_auroc = 0.
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # === Train ===
        model.train()
        total_loss = 0.0
        all_node_scores = []
        all_edge_scores = []
        for batch in tqdm(train_loader, desc=f"Train {epoch}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)

            node_output = output['node_output']
            edge_output = output['edge_output']
            nll_g, nll_v, sigma_g_avg, sigma_v_avg, node_score = node_output
            nll, sigma_avg = edge_output
            edge_to_node_score = score_aggregate(batch.edge_index, nll, node_score.size(0), 'topk')
            all_node_scores.append(node_score.detach().cpu())
            all_edge_scores.append(edge_to_node_score.detach().cpu())

            losses = unsup_loss(output['node_output'], output['edge_output'], batch_index=batch.batch, edge_batch_index=batch.batch[batch.edge_index[0]])
            loss = losses['loss_total']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.6f}")

        with torch.no_grad():
            # Evaluate
            all_node_scores = torch.cat(all_node_scores)
            all_edge_scores = torch.cat(all_edge_scores)
            mu_node, sigma_node = all_node_scores.mean(), all_node_scores.std()
            mu_edge, sigma_edge = all_edge_scores.mean(), all_edge_scores.std()
            m = 0.8 if epoch <= 5 else 0.9
            ema_update(model.mu_node, mu_node, m); ema_update(model.sigma_node, sigma_node, m)
            ema_update(model.mu_edge, mu_edge, m); ema_update(model.sigma_edge, sigma_edge, m)

            # == Test ==
            model.eval()
            instance_scores, instance_labels = [], []
            pixel_scores, pixel_labels = [], []
            anomaly_pixel_scores, anomaly_pixel_labels = [], []
            pixel_label_maps, pixel_score_maps = [], []
            all_obj_scores, all_obj_labels = [], []
            top1_hits = []
            for batch in tqdm(val_loader, desc=f"Val {epoch}"):
                batch = batch.to(device)
                output = model(batch)
                anomaly_scores = output['score']
                node_batch = batch.batch
                num_graphs = len(batch.label)

                # Object AUROC
                all_obj_scores.append(output['score'].detach().cpu().numpy())
                all_obj_labels.append(batch.node_labels.detach().cpu().numpy())

                for i in range(num_graphs):
                    mask = (node_batch == i)
                    node_indices = mask.nonzero(as_tuple=True)[0]
                    node_scores_i = anomaly_scores[node_indices]

                    thing_mask = batch.is_thing[node_indices].to(torch.bool)
                    if not thing_mask.any():
                        # instance_scores.append(0.0)
                        # instance_labels.append(int(batch.label[i].item()))
                        continue  

                    node_scores_i = node_scores_i[thing_mask]
                    node_labels_i = batch.node_labels[node_indices][thing_mask].to(torch.bool)
                    node_masks_i = [m for m, is_t in zip(batch.node_mask[i], thing_mask) if is_t]

                    # Top k:
                    if int(batch.label[i].item()) != 0:
                        K = 10
                        topk = min(K, node_scores_i.numel())
                        topk_idx = torch.topk(node_scores_i, topk).indices
                        hit = int(node_labels_i[topk_idx].any().item())
                        top1_hits.append(hit)

                    # top-quantile or max
                    N = node_scores_i.numel()
                    if N == 0:
                        instance_score = 0.0
                    else:
                        k = max(1, min(int(0.05 * N), 10))
                        instance_score = torch.topk(node_scores_i, k).values.mean().item()
                    instance_scores.append(instance_score)
                    instance_labels.append(int(batch.label[i].item()))

                    # pixel-level
                    H, W = batch.image_size[i]
                    node_masks = batch.node_mask[i]
                    pixel_score_map = build_pixel_score_map(
                        node_masks=node_masks_i,
                        node_scores=node_scores_i.detach().cpu().numpy(),
                        H=H, W=W,
                        mapping="soft",  # "hard" 或 "soft"
                        sigma=5.0
                    )
                    if i == 0:
                        pixel_score_maps.append(pixel_score_map)
                    pixel_scores.append(pixel_score_map.flatten())

                    # gt mask
                    label = int(batch.label[i].item())
                    if label == 1:
                        gt_mask_path = batch.gt_mask[i]
                        if not os.path.exists(gt_mask_path):
                            continue

                        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                        gt_mask = (gt_mask > 0).astype(bool)
                        if gt_mask.shape != (H, W):
                            gt_mask = cv2.resize(gt_mask.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST) > 0

                        if i == 0:
                            pixel_label_maps.append(gt_mask)
                        pixel_labels.append(gt_mask.flatten())
                        anomaly_pixel_scores.append(pixel_score_map.flatten())
                        anomaly_pixel_labels.append(gt_mask.flatten())
                    else:
                        zero_mask = np.zeros((H, W), dtype=bool)
                        if i == 0:
                            pixel_label_maps.append(zero_mask)
                        pixel_labels.append(zero_mask.flatten())

        metrics = evaluate_metrics(
            instance_labels=np.array(instance_labels),
            instance_scores=np.array(instance_scores),
            pixel_labels=np.concatenate(pixel_labels),
            pixel_scores=np.concatenate(pixel_scores),
            anomaly_pixel_labels=anomaly_pixel_labels,
            anomaly_pixel_scores=anomaly_pixel_scores,
            pixel_label_maps=pixel_label_maps,
            pixel_score_maps=pixel_score_maps,
            object_scores=np.concatenate(all_obj_scores) if len(all_obj_scores) else np.array([]),
            object_labels=np.concatenate(all_obj_labels) if len(all_obj_labels) else np.array([])
        )

        top1_hit_rate = float(np.mean(top1_hits)) if top1_hits else float('nan')
        print(f"Anomalous-image Top1-hit (thing-only): {top1_hit_rate:.3f}")
        
        scheduler.step(metrics["instance_auroc"])
        print(
            f"[Val {epoch}] "
            f"Instance AUROC={metrics['instance_auroc']:.4f} | "
            f"Full Pixel AUROC={metrics['full_pixel_auroc']:.4f} | "
            f"Anomaly Pixel AUROC={metrics['anomaly_pixel_auroc']:.4f} | "
            f"I-AP={metrics['instance_ap']:.4f} | "
            f"P-AP={metrics['pixel_ap']:.4f} | "
            f"FPR={metrics['pixel_fpr95']:.4f} | "
            f"Object AUROC={metrics['object_auroc']:.4f} | "
            f"Object AP={metrics['object_ap']:.4f} | "
            f"Object FPR={metrics['object_fpr95']:.4f}\n"
        )
                
        # == log ==
        row = {
            "epoch": epoch,
            "loss": total_loss,
            "i-auroc": metrics["instance_auroc"],
            "p-auroc": metrics["full_pixel_auroc"],
            "aupro": metrics["anomaly_pixel_auroc"],
            'fpr': metrics['pixel_fpr95'],
            'i-ap': metrics['instance_ap'],
            'p-ap': metrics['pixel_ap'],
            "o-auroc": metrics["object_auroc"],
            "o-ap": metrics['object_ap'],
            "o-fpr": metrics['object_fpr95'],
            'top1-hit': top1_hit_rate
        }

        df = pd.DataFrame([row])
        df.to_csv(csv_path, index=False, mode="a",
                  header=not os.path.exists(csv_path), float_format="%.6f")

        # save ckpt
        save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }, save_path)

    print(f"\nTraining completed. Models saved at: {save_dir}")
    print(f"Metrics saved to {csv_path}")
    plot_metrics(os.path.join(save_dir, "metrics.csv"), save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LayoutAD Training Script")

    parser.add_argument("--data_root", type=str, default='/home/zengzc/Projects/datasets/LAD', help="Path to dataset root")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers")

    parser.add_argument("--d_model", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--num_enc_layers", type=int, default=8, help="Number of Transformer encoder layers")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--rel_dim", type=int, default=32, help="Edge relation dimension")

    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda:2", help="Training device (e.g., cuda:0)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)