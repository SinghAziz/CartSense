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
