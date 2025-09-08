#!/usr/bin/env python3
# SlimSAM/eval_sa1b.py
# Evaluate SlimSAM / SAM on SA-1B-style (image + per-image JSON) dataset.
# Supports:
#   - auto_segment : AutomaticMaskGenerator (no prompt) → mean_best_iou, Prec/Rec@τ
#   - miou_point   : SamPredictor(one positive point per GT) → mIoU
# Also visualizes top/bottom n% triptychs [Original | GT | Pred].

# =============== All imports at top ===============
import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

# pycocotools for mask encode/decode + IoU
from pycocotools import mask as mask_utils

# SlimSAM repo ships a SAM fork under segment_anything/
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
    SamPredictor,
)

import matplotlib.pyplot as plt

from knockknock import discord_sender
from torch_pruning.utils import count_ops_and_params
import sys
from PTQ import (
    MinMaxActWrapper, MinMaxActFakeQuant, MinMaxActObserver,
    StaticQuantWrapper, FakeQuantize, MinMaxObserver
)
from dotenv import load_dotenv

# 저장 당시 기록된 심볼 경로(__main__.*)를 현재 모듈로 매핑
sys.modules['__main__'].MinMaxActWrapper = MinMaxActWrapper
sys.modules['__main__'].MinMaxActFakeQuant = MinMaxActFakeQuant
sys.modules['__main__'].MinMaxActObserver = MinMaxActObserver
sys.modules['__main__'].StaticQuantWrapper = StaticQuantWrapper
sys.modules['__main__'].FakeQuantize = FakeQuantize
sys.modules['__main__'].MinMaxObserver = MinMaxObserver
# ==================================================

# Load environment variables (expect .env at project root). Missing .env is okay.
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def _noop_decorator(func):
    return func

_DECORATOR = discord_sender(webhook_url=DISCORD_WEBHOOK_URL) if DISCORD_WEBHOOK_URL else _noop_decorator

def fix_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# ----------------- Mask utils -----------------
def binmask_to_rle(binmask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(binmask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def rle_to_binmask(rle: Dict[str, Any]) -> np.ndarray:
    return mask_utils.decode(rle).astype(bool)

def iou_matrix(gt_rles: List[Dict], pred_rles: List[Dict]) -> np.ndarray:
    if len(gt_rles) == 0 or len(pred_rles) == 0:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=float)
    # pycocotools.iou expects (dt=preds, gt), returns (len(dt), len(gt))
    ious = mask_utils.iou(dt=pred_rles, gt=gt_rles, pyiscrowd=[0] * len(gt_rles))
    return ious.T  # (N_gt, M_pred)

def rles_to_stack(rles: List[Dict], h: int, w: int) -> np.ndarray:
    """Decode list of RLEs to boolean stack [N, H, W]."""
    if len(rles) == 0:
        return np.zeros((0, h, w), dtype=bool)
    ms = []
    for r in rles:
        m = rle_to_binmask(r)
        if m.shape == (h, w):
            ms.append(m.astype(bool))
    if not ms:
        return np.zeros((0, h, w), dtype=bool)
    return np.stack(ms, axis=0)

def overlay_masks_on_image(img_np: np.ndarray, masks_stack: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    img_np: [H, W, 3] uint8
    masks_stack: [N, H, W] bool
    """
    if masks_stack.size == 0:
        return img_np
    out = img_np.astype(np.float32).copy()
    rng = np.random.RandomState(1234)  # deterministic colors
    colors = rng.randint(0, 255, size=(masks_stack.shape[0], 3), dtype=np.uint8).astype(np.float32)
    for i, m in enumerate(masks_stack):
        if not m.any():
            continue
        c = colors[i]  # (3,)
        out[m] = (1.0 - alpha) * out[m] + alpha * c
    return out.clip(0, 255).astype(np.uint8)

def save_triptych(img_path: Path, gt_rles: List[Dict], pred_rles: List[Dict],
                  width: Optional[int], height: Optional[int], save_path: Path):
    """Save [Original | GT overlay | Pred overlay] as one image."""
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
    except Exception:
        return
    if width and height and (width, height) != (w0, h0):
        im = im.resize((width, height), Image.BILINEAR)
        w, h = width, height
    else:
        w, h = im.size
    img_np = np.array(im)
    gt_stack   = rles_to_stack(gt_rles,   h, w)
    pred_stack = rles_to_stack(pred_rles, h, w)
    gt_overlay   = overlay_masks_on_image(img_np, gt_stack, alpha=0.5)
    pred_overlay = overlay_masks_on_image(img_np, pred_stack, alpha=0.5)

    fig = plt.figure(figsize=(12, 4), dpi=150)
    ax1 = fig.add_subplot(1, 3, 1); ax1.imshow(img_np);       ax1.set_title("Original");   ax1.axis("off")
    ax2 = fig.add_subplot(1, 3, 2); ax2.imshow(gt_overlay);   ax2.set_title("GT");         ax2.axis("off")
    ax3 = fig.add_subplot(1, 3, 3); ax3.imshow(pred_overlay); ax3.set_title("Prediction"); ax3.axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


# ----------------- SA-1B (image + per-image JSON) -----------------
class SA1BFolderDataset:
    """
    Expect:
        root/
          abc.jpg
          abc.json
          def.jpg
          def.json
          ...
    JSON: list or {'annotations': [...]} with each ann having 'segmentation' (RLE or polygon).
    """
    def __init__(self, root: Path, exts=(".jpg", ".jpeg", ".png"), max_images: Optional[int] = None):
        self.root = Path(root)
        pairs = []
        for p in sorted(self.root.iterdir()):
            if p.suffix.lower() in exts:
                j = p.with_suffix(".json")
                if j.exists():
                    pairs.append((p, j))
        if max_images:
            pairs = pairs[:max_images]
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for img_path, json_path in self.pairs:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "annotations" in data:
                anns = data["annotations"]
            elif isinstance(data, list):
                anns = data
            else:
                anns = []

            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                w = h = None

            gt_rles = []
            for ann in anns:
                seg = ann.get("segmentation", None)
                if seg is None:
                    continue
                if isinstance(seg, dict) and "counts" in seg:
                    rle = dict(seg)
                    if isinstance(rle["counts"], bytes):
                        rle["counts"] = rle["counts"].decode("ascii")
                    gt_rles.append(rle)
                elif isinstance(seg, list) and w is not None and h is not None:
                    rles = mask_utils.frPyObjects(seg, h, w)
                    rle = mask_utils.merge(rles)
                    if isinstance(rle["counts"], bytes):
                        rle["counts"] = rle["counts"].decode("ascii")
                    gt_rles.append(rle)

            yield {
                "file": img_path,
                "json": json_path,
                "width": w,
                "height": h,
                "gt_rles": gt_rles,
            }


# ----------------- Point helpers (for mIoU) -----------------
def rle_centroid_point(rle: Dict[str, Any]) -> Tuple[int, int]:
    m = mask_utils.decode(rle).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return 0, 0
    cx = int(xs.mean()); cy = int(ys.mean())
    return cx, cy  # (x, y)

def extract_single_point_from_ann(ann: Dict[str, Any], fallback_rle: Dict[str, Any]) -> Tuple[int, int]:
    # 1) try known keys
    for k in ["point_coords", "point", "pos_point", "positive_point"]:
        if k in ann and isinstance(ann[k], (list, tuple)) and len(ann[k]) >= 2:
            x, y = ann[k][0], ann[k][1]
            try:
                return int(round(float(x))), int(round(float(y)))
            except Exception:
                pass
    # 2) fallback: centroid of GT mask
    return rle_centroid_point(fallback_rle)


# ----------------- Model wrappers -----------------
class BaseSegModel:
    def predict_masks(self, image: Image.Image) -> List[Dict[str, Any]]:
        raise NotImplementedError

# (A) Auto-segmentation wrappers (no prompt)
class SAMAutoWrapper(BaseSegModel):
    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_h",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        crop_n_layers: int = 1,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 0,
        device: str = "cuda",
    ):
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device).eval()
        self.automask = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )

    def predict_masks(self, image: Image.Image) -> List[Dict[str, Any]]:
        np_img = np.array(image.convert("RGB"))
        results = self.automask.generate(np_img)
        out = []
        for r in results:
            rle = binmask_to_rle(r["segmentation"])
            score = float(r.get("predicted_iou", 0.0))
            out.append({"rle": rle, "score": score})
        return out

class SlimSAMUniformAutoWrapper(BaseSegModel):
    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_p50",  # vit_p50 or vit_p77
        points_per_side: int = 32,
        device: str = "cuda",
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        crop_n_layers: int = 1,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 0,
    ):
        model = sam_model_registry[model_type](checkpoint=checkpoint)
        model.to(device).eval()
        self.automask = SamAutomaticMaskGenerator(
            model=model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )

    def predict_masks(self, image: Image.Image) -> List[Dict[str, Any]]:
        np_img = np.array(image.convert("RGB"))
        results = self.automask.generate(np_img)
        out = []
        for r in results:
            rle = binmask_to_rle(r["segmentation"])
            score = float(r.get("predicted_iou", 0.0))
            out.append({"rle": rle, "score": score})
        return out

class SlimSAMGlobalAutoWrapper(BaseSegModel):
    def __init__(
        self,
        checkpoint: str,
        points_per_side: int = 32,
        device: str = "cuda",
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        crop_n_layers: int = 1,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 0,
    ):
        model = torch.load(checkpoint, map_location=device, weights_only=False)
        if hasattr(model.image_encoder, "module"):
            model.image_encoder = model.image_encoder.module

        import types
        def patched_forward(self, x):
            x = self.patch_embed(x)
            if getattr(self, "pos_embed", None) is not None:
                x = x + self.pos_embed
            for blk in self.blocks:
                x, qkv_emb, mid_emb, x_emb = blk(x)
            x = self.neck(x.permute(0, 3, 1, 2))
            return x
        model.image_encoder.forward = types.MethodType(patched_forward, model.image_encoder)

        model.to(device).eval()
        self.automask = SamAutomaticMaskGenerator(
            model=model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )

    def predict_masks(self, image: Image.Image) -> List[Dict[str, Any]]:
        np_img = np.array(image.convert("RGB"))
        results = self.automask.generate(np_img)
        out = []
        for r in results:
            rle = binmask_to_rle(r["segmentation"])
            score = float(r.get("predicted_iou", 0.0))
            out.append({"rle": rle, "score": score})
        return out


# (B) Predictor wrappers (single-point prompt → one mask)
class SAMPredictorWrapper:
    def __init__(self, checkpoint: str, model_type: str, device: str = "cuda"):
        model = sam_model_registry[model_type](checkpoint=checkpoint)
        model.to(device).eval()
        self.predictor = SamPredictor(model)

    def set_image(self, np_img: np.ndarray):
        self.predictor.set_image(np_img)

    def predict_one_mask(self, point_xy: Tuple[int, int]) -> Dict[str, Any]:
        # set_image는 외부(이미지 루프에서) 단 1회 호출!
        points = np.array([point_xy], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        with torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=points, point_labels=labels, multimask_output=False
            )
        binmask = masks[0].astype(bool)
        return {"rle": binmask_to_rle(binmask), "score": float(scores[0])}

class SlimSAMGlobalPredictorWrapper:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        model = torch.load(checkpoint, map_location=device, weights_only=False)
        if hasattr(model.image_encoder, "module"):
            model.image_encoder = model.image_encoder.module
        import types
        def patched_forward(self, x):
            x = self.patch_embed(x)
            if getattr(self, "pos_embed", None) is not None:
                x = x + self.pos_embed
            for blk in self.blocks:
                x, qkv_emb, mid_emb, x_emb = blk(x)
            x = self.neck(x.permute(0, 3, 1, 2))
            return x
        model.image_encoder.forward = types.MethodType(patched_forward, model.image_encoder)
        model.to(device).eval()
        self.predictor = SamPredictor(model)

    def set_image(self, np_img: np.ndarray):
        self.predictor.set_image(np_img)

    def predict_one_mask(self, point_xy: Tuple[int, int]) -> Dict[str, Any]:
        points = np.array([point_xy], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        with torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=points, point_labels=labels, multimask_output=False
            )
        binmask = masks[0].astype(bool)
        return {"rle": binmask_to_rle(binmask), "score": float(scores[0])}


# ----------------- Greedy matching / PR (auto_segment mode) -----------------
def greedy_match(ious: np.ndarray, thr: float) -> Tuple[int, int, int]:
    if ious.size == 0:
        return 0, ious.shape[1], ious.shape[0]
    N, M = ious.shape
    gt_used = np.zeros(N, dtype=bool)
    pr_used = np.zeros(M, dtype=bool)
    pairs: List[Tuple[int, int, float]] = [
        (g, p, ious[g, p]) for g in range(N) for p in range(M) if ious[g, p] >= thr
    ]
    pairs.sort(key=lambda x: x[2], reverse=True)
    tp = 0
    for g, p, _ in pairs:
        if not gt_used[g] and not pr_used[p]:
            gt_used[g] = True
            pr_used[p] = True
            tp += 1
    fp = (~pr_used).sum()
    fn = (~gt_used).sum()
    return tp, fp, fn

def precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec


# ----------------- Evaluation loop -----------------
def evaluate(
    data_root: Path,
    model_kind: str,
    ckpt_path: str,
    out_dir: Path,
    device: str = "cuda",
    max_images: Optional[int] = None,
    points_per_side: int = 32,
    iou_thrs: List[float] = [0.50, 0.75, 0.90],
    sam_model_type: str = "vit_h",     # for --model sam (auto or predictor)
    slimsam_uniform_type: str = "vit_p50",  # 'vit_p50' or 'vit_p77'
    eval_mode: str = "auto_segment",   # 'auto_segment' or 'miou_point'
    viz_percent: float = 0.0,
    viz_metric: str = "mean_best_iou", # 'auto_segment': mean_best_iou/prec@..., 'miou_point': miou
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = SA1BFolderDataset(root=data_root, max_images=max_images)

    # Build model wrapper(s)
    if eval_mode == "auto_segment":
        if model_kind == "sam":
            model = SAMAutoWrapper(
                checkpoint=ckpt_path,
                model_type=sam_model_type,
                points_per_side=points_per_side,
                device=device,
            )
        elif model_kind == "slimsam_uniform":
            model = SlimSAMUniformAutoWrapper(
                checkpoint=ckpt_path,
                model_type=slimsam_uniform_type,
                points_per_side=points_per_side,
                device=device,
            )
        elif model_kind == "slimsam_global":
            model = SlimSAMGlobalAutoWrapper(
                checkpoint=ckpt_path,
                points_per_side=points_per_side,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported model_kind: {model_kind}")
    else:  # miou_point
        if model_kind == "sam":
            predictor = SAMPredictorWrapper(ckpt_path, sam_model_type, device)
        elif model_kind == "slimsam_uniform":
            predictor = SAMPredictorWrapper(ckpt_path, slimsam_uniform_type, device)
        elif model_kind == "slimsam_global":
            predictor = SlimSAMGlobalPredictorWrapper(ckpt_path, device)
        else:
            raise ValueError(f"Unsupported model_kind: {model_kind}")
        
        # For miou_point mode, get model from predictor
        model = predictor.predictor.model
        
    # Get Model Size
    model.image_encoder.eval()
    example_inputs = torch.randn(1, 3, 1024, 1024).to(device)
    ori_macs, ori_size = count_ops_and_params(model.image_encoder, example_inputs)
    print("MACs(G):",ori_macs/1e9,"Params(M):",ori_size/1e6)

    # Aggregates & caches
    per_image_rows: List[Dict[str, Any]] = []
    viz_cache: List[Dict[str, Any]] = []

    if eval_mode == "auto_segment":
        totals = {thr: {"prec": 0.0, "rec": 0.0, "n": 0} for thr in iou_thrs}
        sum_mean_best = 0.0
        num_with_gt = 0
        num_eval = 0

        for sample in tqdm(ds, desc="Evaluating (auto_segment)"):
            img_path: Path = sample["file"]
            gt_rles: List[Dict] = sample["gt_rles"]
            width, height = sample["width"], sample["height"]

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            preds = model.predict_masks(image)
            pred_rles = [p["rle"] for p in preds]

            ious = iou_matrix(gt_rles, pred_rles)  # (N_gt, M_pred)

            if len(gt_rles) > 0:
                best_per_gt = ious.max(axis=1) if ious.size > 0 else np.zeros((len(gt_rles),), dtype=float)
                mean_best = float(best_per_gt.mean()) if best_per_gt.size > 0 else 0.0
                sum_mean_best += mean_best
                num_with_gt += 1
            else:
                mean_best = 0.0

            pr_this = {}
            for thr in iou_thrs:
                tp, fp, fn = greedy_match(ious, thr)
                prec, rec = precision_recall(tp, fp, fn)
                totals[thr]["prec"] += prec
                totals[thr]["rec"] += rec
                totals[thr]["n"] += 1
                pr_this[f"prec@{thr:.2f}"] = round(prec, 4)
                pr_this[f"rec@{thr:.2f}"]  = round(rec, 4)

            row = {
                "file": img_path.name,
                "num_gt": len(gt_rles),
                "num_pred": len(pred_rles),
                "mean_best_iou": round(mean_best, 4),
                **pr_this,
            }
            per_image_rows.append(row)
            num_eval += 1

            if viz_percent > 0:
                viz_cache.append({
                    "file_path": img_path,
                    "width": width,
                    "height": height,
                    "gt_rles": gt_rles,
                    "pred_rles": pred_rles,
                    "metrics": row,
                })

        # Write per-image CSV
        per_image_csv = out_dir / "per_image_metrics.csv"
        if per_image_rows:
            with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_image_rows)

        mean_best = (sum_mean_best / num_with_gt) if num_with_gt > 0 else 0.0
        avg_prec = {thr: (totals[thr]["prec"] / max(totals[thr]["n"], 1)) for thr in iou_thrs}
        avg_rec  = {thr: (totals[thr]["rec"]  / max(totals[thr]["n"], 1)) for thr in iou_thrs}

        print("\n===== Evaluation Summary (auto_segment) =====")
        print(f"MACs                 : {ori_macs/1e9}G")
        print(f"Params               : {ori_size/1e6}M")
        print(f"Images evaluated     : {num_eval}")
        print(f"Images with GT       : {num_with_gt}")
        print(f"Mean BestIoU (GTwise): {mean_best:.4f}")
        for thr in iou_thrs:
            print(f"Prec@{thr:.2f} / Rec@{thr:.2f}: {avg_prec[thr]:.4f} / {avg_rec[thr]:.4f}")

        summary_csv = out_dir / "summary_metrics.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["MACs(G)", ori_macs/1e9])
            w.writerow(["Params(M)", ori_size/1e6])
            w.writerow(["num_images", num_eval])
            w.writerow(["num_images_with_gt", num_with_gt])
            w.writerow(["mean_best_iou", round(mean_best, 4)])
            for thr in iou_thrs:
                w.writerow([f"prec@{thr:.2f}", round(avg_prec[thr], 4)])
                w.writerow([f"rec@{thr:.2f}",  round(avg_rec[thr],  4)])

    else:  # miou_point
        total_iou_sum = 0.0
        total_iou_cnt = 0
        num_eval = 0

        for sample in tqdm(ds, desc="Evaluating (miou_point)"):
            img_path: Path = sample["file"]
            gt_rles: List[Dict] = sample["gt_rles"]
            width, height = sample["width"], sample["height"]

            try:
                image = Image.open(img_path).convert("RGB")
                np_img = np.array(image)
                predictor.set_image(np_img)
            except Exception:
                continue

            # load per-image annotations to read point (if available)
            with open(sample["json"], "r", encoding="utf-8") as f:
                data = json.load(f)
            anns = data["annotations"] if isinstance(data, dict) and "annotations" in data else (data if isinstance(data, list) else [])

            ious_this = []
            pred_rles_this = []
            for ann, rle in zip(anns, gt_rles):
                px, py = extract_single_point_from_ann(ann, rle)
                pred = predictor.predict_one_mask((px, py))  # set_image 재호출 없음
                pred_rles_this.append(pred["rle"])
                iou = mask_utils.iou([pred["rle"]], [rle], [0])[0, 0]
                ious_this.append(float(iou))
                total_iou_sum += float(iou)
                total_iou_cnt += 1

            miou_img = float(np.mean(ious_this)) if ious_this else 0.0
            row = {"file": img_path.name, "num_gt": len(gt_rles), "miou": round(miou_img, 4)}
            per_image_rows.append(row)
            num_eval += 1

            if viz_percent > 0:
                viz_cache.append({
                    "file_path": img_path,
                    "width": width,
                    "height": height,
                    "gt_rles": gt_rles,
                    "pred_rles": pred_rles_this,  # union of per-GT predicted masks
                    "metrics": row,
                })

        # Write per-image CSV
        per_image_csv = out_dir / "per_image_metrics.csv"
        if per_image_rows:
            with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_image_rows)

        dataset_miou = (total_iou_sum / max(total_iou_cnt, 1))
        print("\n===== Evaluation Summary (mIoU, point-prompt) =====")
        print(f"MACs                 : {ori_macs/1e9}G")
        print(f"Params               : {ori_size/1e6}M")
        print(f"Images evaluated   : {num_eval}")
        print(f"Total GT instances : {total_iou_cnt}")
        print(f"Dataset mIoU       : {dataset_miou:.4f}")

        summary_csv = out_dir / "summary_metrics.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["MACs(G)", ori_macs/1e9])
            w.writerow(["Params(M)", ori_size/1e6])
            w.writerow(["num_images", num_eval])
            w.writerow(["num_gt_instances", total_iou_cnt])
            w.writerow(["dataset_miou", round(dataset_miou, 4)])

    # ---------- Visualization (top/bottom n%) ----------
    if viz_percent > 0 and len(viz_cache) > 0:
        metric_key = viz_metric
        if metric_key not in viz_cache[0]["metrics"]:
            # attempt fallback for common cases
            if eval_mode == "miou_point":
                metric_key = "miou"
            else:
                metric_key = "mean_best_iou"
            print(f"[viz] WARNING: metric not found; fallback to '{metric_key}'.")

        k = max(1, int(math.ceil(len(viz_cache) * (viz_percent / 100.0))))
        print(f"[viz] Selecting top {k} and bottom {k} by '{metric_key}' (from {len(viz_cache)} images).")

        sorted_items = sorted(viz_cache, key=lambda s: float(s["metrics"][metric_key]))
        bottom = sorted_items[:k]
        top    = sorted_items[-k:]

        top_dir = out_dir / "viz" / "top"
        bot_dir = out_dir / "viz" / "bottom"

        for rank, s in enumerate(top[::-1], 1):  # highest first
            save_path = top_dir / f"{rank:03d}_{s['file_path'].stem}_{metric_key}_{s['metrics'][metric_key]}.png"
            save_triptych(s["file_path"], s["gt_rles"], s["pred_rles"], s["width"], s["height"], save_path)

        for rank, s in enumerate(bottom, 1):     # lowest first
            save_path = bot_dir / f"{rank:03d}_{s['file_path'].stem}_{metric_key}_{s['metrics'][metric_key]}.png"
            save_triptych(s["file_path"], s["gt_rles"], s["pred_rles"], s["width"], s["height"], save_path)

        print(f"[viz] Saved top triptychs to   : {top_dir}")
        print(f"[viz] Saved bottom triptychs to: {bot_dir}")

    return per_image_rows


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate SlimSAM / SAM on SA-1B (image+json) with auto-seg or single-point mIoU; visualize top/bottom n%."
    )
    ap.add_argument("--data_root", type=str, required=True, help="Folder containing *.jpg and *.json pairs")
    ap.add_argument("--model", type=str, required=True,
                    choices=["sam", "slimsam_uniform", "slimsam_global"])
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write CSV/viz")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--points_per_side", type=int, default=32)
    ap.add_argument("--iou_thrs", type=str, default="0.50,0.75",
                    help="Comma-separated IoU thresholds (auto_segment mode).")
    ap.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--slimsam_uniform_type", type=str, default="vit_p50", choices=["vit_p50", "vit_p77"])
    ap.add_argument("--eval_mode", type=str, default="auto_segment",
                    choices=["auto_segment", "miou_point"],
                    help="Use 'miou_point' to reproduce SlimSAM mIoU (single positive point per GT).")
    # visualization
    ap.add_argument("--viz_percent", type=float, default=0.0,
                    help="Percent (0-100) to visualize for top and bottom groups.")
    ap.add_argument("--viz_metric", type=str, default="mean_best_iou",
                    help="Ranking metric: auto_segment -> e.g., 'mean_best_iou' or 'prec@0.50'; miou_point -> 'miou'.")
    return ap.parse_args()

@_DECORATOR
def main():
    fix_seed()
    args = parse_args()
    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    iou_thrs = [float(x.strip()) for x in args.iou_thrs.split(",") if x.strip()]
    evaluate(
        data_root=root,
        model_kind=args.model,
        ckpt_path=args.ckpt,
        out_dir=out_dir,
        device=args.device,
        max_images=args.max_images,
        points_per_side=args.points_per_side,
        iou_thrs=iou_thrs,
        sam_model_type=args.sam_model_type,
        slimsam_uniform_type=args.slimsam_uniform_type,
        eval_mode=args.eval_mode,
        viz_percent=args.viz_percent,
        viz_metric=args.viz_metric,
    )

if __name__ == "__main__":
    main()
