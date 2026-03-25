import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_curve
)
from skimage.measure import label, regionprops


def compute_aupro(gt_mask, score_map, max_fpr=0.3, num_th=100):
    if gt_mask.sum() == 0:
        return 0.0
    thresholds = np.linspace(score_map.min(), score_map.max(), num_th)
    fprs, pros = [], []
    labeled_mask = label(gt_mask)
    regions = regionprops(labeled_mask)
    for t in thresholds:
        pred_mask = (score_map > t).astype(np.uint8)
        fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
        tn = (gt_mask == 0).sum()
        fpr = fp / (tn + 1e-8)
        if fpr > max_fpr:
            continue
        region_ious = []
        for reg in regions:
            region_mask = (labeled_mask == reg.label)
            inter = np.logical_and(pred_mask, region_mask).sum()
            region_ious.append(np.clip(inter / (region_mask.sum() + 1e-8), 0, 1))
        fprs.append(fpr)
        pros.append(np.mean(region_ious) if region_ious else 0.0)
    if len(fprs) < 2:
        return 0.0
    fprs, pros = np.array(fprs), np.array(pros)
    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]
    unique, idx = np.unique(fprs, return_index=True)
    fprs, pros = fprs[idx], np.clip(pros[idx], 0, 1)
    return float(np.trapz(pros, fprs) / (max_fpr + 1e-8))

def compute_fpr_at_tpr(labels, scores, target_tpr=0.95):

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(labels)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fpr95 = float(np.interp(target_tpr, tpr, fpr))
    return max(0.0, min(1.0, fpr95))

def evaluate_metrics(
    instance_labels,
    instance_scores,
    pixel_labels,
    pixel_scores,
    object_scores,
    object_labels,
    anomaly_pixel_labels=None,
    anomaly_pixel_scores=None,
    pixel_label_maps=None,
    pixel_score_maps=None,
):

    metrics = {}

    import time
    start = time.time()

    # Instance-level AUROC
    if len(np.unique(instance_labels)) > 1:
        metrics["instance_auroc"] = roc_auc_score(instance_labels, instance_scores)
    else:
        metrics["instance_auroc"] = 0.0
    print(metrics["instance_auroc"], f"{(time.time()-start):.2f}s")

    # Full-pixel AUROC
    if len(np.unique(pixel_labels)) > 1:
        metrics["full_pixel_auroc"] = roc_auc_score(pixel_labels, pixel_scores)
    else:
        metrics["full_pixel_auroc"] = 0.0
    print(metrics["full_pixel_auroc"], f"{(time.time()-start):.2f}s")

    # Anomaly-pixel AUROC
    if anomaly_pixel_scores is not None and len(anomaly_pixel_scores) > 0:
        anom_labels = np.concatenate(anomaly_pixel_labels) if isinstance(anomaly_pixel_labels, list) else anomaly_pixel_labels
        anom_scores = np.concatenate(anomaly_pixel_scores) if isinstance(anomaly_pixel_scores, list) else anomaly_pixel_scores
        metrics["anomaly_pixel_auroc"] = roc_auc_score(anom_labels, anom_scores)
    else:
        metrics["anomaly_pixel_auroc"] = 0.0
    print(metrics["anomaly_pixel_auroc"], f"{(time.time()-start):.2f}s")

    # instance-AP
    if len(np.unique(instance_labels)) > 1:
        metrics["instance_ap"] = float(average_precision_score(instance_labels, instance_scores))
    else:
        metrics["instance_ap"] = 0.0
    print(metrics["instance_ap"], f"{(time.time()-start):.2f}s")

    # pixel-ap
    if len(np.unique(pixel_labels)) > 1:
        metrics["pixel_ap"] = float(average_precision_score(pixel_labels, pixel_scores))
    else:
        metrics["pixel_ap"] = 0.0
    print(metrics["pixel_ap"], f"{(time.time()-start):.2f}s")

    # FPR@95%TPR
    if len(np.unique(pixel_labels)) > 1:
        metrics["pixel_fpr95"] = compute_fpr_at_tpr(pixel_labels, pixel_scores, target_tpr=0.95)
    else:
        metrics["pixel_fpr95"] = 0.0
    print(metrics["pixel_fpr95"], f"{(time.time()-start):.2f}s")

    # Object AUROC
    if object_scores is not None and len(object_scores) > 0 and len(np.unique(object_labels)) > 1:
        metrics["object_auroc"] = float(roc_auc_score(object_labels, object_scores))
        metrics["object_ap"]    = float(average_precision_score(object_labels, object_scores))
        metrics["object_fpr95"] = compute_fpr_at_tpr(object_labels, object_scores, target_tpr=0.95)
    else:
        metrics["object_auroc"] = 0.0
        metrics["object_ap"]    = 0.0
        metrics["object_fpr95"] = 0.0
    print(metrics["object_auroc"], f"{(time.time()-start):.2f}s")
    print(metrics["object_ap"], f"{(time.time()-start):.2f}s")
    print(metrics["object_fpr95"], f"{(time.time()-start):.2f}s")

    return metrics
