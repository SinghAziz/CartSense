"""CartSense-specific metrics only. For detection mAP use pycocotools/ultralytics val on a labelled set (none available here)."""
import numpy as np

def latency_stats(ms):
    if not ms: return {}
    a = np.array(ms); return dict(n=len(a), median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)), mean_ms=float(a.mean()))

def event_metrics(pred, gt, tol_s=1.0):
    """pred/gt: lists of dict(type, sku, t). One-to-one greedy match on (type, sku) within tol_s."""
    used, tp = set(), 0
    for g in gt:
        for i, p in enumerate(pred):
            if i not in used and p["type"] == g["type"] and p["sku"] == g["sku"] and abs(p["t"] - g["t"]) <= tol_s:
                used.add(i); tp += 1; break
    prec = tp / len(pred) if pred else 0.0; rec = tp / len(gt) if gt else 0.0
    return dict(tp=tp, fp=len(pred) - tp, fn=len(gt) - tp, precision=prec, recall=rec)

def cart_accuracy(pred_items, gt_items):
    return dict(exact_match=pred_items == gt_items,
                sku_diffs={k: (pred_items.get(k, 0), gt_items.get(k, 0)) for k in set(pred_items) | set(gt_items)
                           if pred_items.get(k, 0) != gt_items.get(k, 0)})

def detection_metrics(pred, gt, iou_threshold=0.5):
    """Class-aware one-to-one precision/recall; mAP requires a labelled dataset evaluator."""
    from src.tracking.geometry import iou
    used = set(); tp = 0
    for g in gt:
        best = max(((iou(p["bbox"], g["bbox"]), i) for i, p in enumerate(pred)
                    if i not in used and p["label"] == g["label"]), default=(0.0, -1))
        if best[0] >= iou_threshold:
            tp += 1; used.add(best[1])
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp,
            "precision": tp / len(pred) if pred else 0.0,
            "recall": tp / len(gt) if gt else 0.0,
            "map": None}

def identification_metrics(pred_skus, gt_skus):
    """Exact SKU metrics; None/unknown predictions count as incorrect."""
    if len(pred_skus) != len(gt_skus):
        raise ValueError("prediction and ground-truth lengths must match")
    labels = sorted(set(gt_skus) | set(pred_skus), key=lambda x: "" if x is None else str(x))
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for p, g in zip(pred_skus, gt_skus):
        matrix.setdefault(g, {}).setdefault(p, 0); matrix[g][p] += 1
    return {"exact_sku_accuracy": sum(p == g for p, g in zip(pred_skus, gt_skus)) / len(gt_skus) if gt_skus else 0.0,
            "confusion_matrix": matrix}

def barcode_metrics(decoded, matched, total):
    return {"decode_success_rate": len(decoded) / total if total else 0.0,
            "catalogue_match_rate": sum(bool(x) for x in matched) / len(matched) if matched else 0.0}
