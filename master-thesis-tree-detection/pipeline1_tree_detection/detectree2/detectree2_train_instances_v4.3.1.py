#!/usr/bin/env python3

"""
detectree2_train_with_iou_complete.py

✅ Detectree2 Training (DefaultTrainer + ValidationLoss + IoU Hook)
✅ Alle wichtigen Parameter (RPN, ROI, Scores optimiert)
✅ Metrics JSON → CSV Konvertierung am Ende
✅ TRAIN + VAL IoU wird berechnet und geloggt!
✅ Alles in EINEM Script
"""

import os
import json
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Polygon
from tqdm import tqdm
import pandas as pd
import numpy as np

from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
import detectron2.utils.comm as comm
import torch

# ===========================================
# 🔧 Konfiguration - OPTIMIERT
# ===========================================

BASE = "/home/abrax/detectree2/training"
SHP_PATH = os.path.join(BASE, "ALL_FINAL.shp")
TILE_DIR = os.path.join(BASE, "images")
COCO_OUT_DIR = os.path.join(BASE, "coco_annotations")
COCO_OUT_JSON = os.path.join(COCO_OUT_DIR, "instances_single_class.json")
OUTPUT_DIR = os.path.join(BASE, "outputs_single_class_with_final")

# Angepasste Parameter für mehr Erkennungen + Stabilität
IMS_PER_BATCH = 1  # Reduziert für GPU-Stabilität
BASE_LR = 0.0005   # Höhere Learning Rate für mehr Aktivierung
MAX_ITER = 3000    # Mehr Iterationen für bessere Konvergenz
NUM_WORKERS = 0
WARMUP_ITERS = 200 # Längere Warmup-Phase
N_EPOCHS = 90

os.makedirs(COCO_OUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("╔═══════════════════════════════════════════════════════════════╗")
print("║  🚀 DETECTREE2 TRAINING + VALIDATION + IoU (COMPLETE)        ║")
print("║     Single-Class Tree Crown Segmentation                     ║")
print("╚═══════════════════════════════════════════════════════════════╝\n")

# ===========================================
# 🔍 Shapefile analysieren - NUR 1 KLASSE
# ===========================================

print("📂 Lade Shapefile für Single-Class Training...")
gdf = gpd.read_file(SHP_PATH)
print(f"✅ {len(gdf)} Kronen geladen.\n")

# IMMER nur eine Klasse: "tree"
cats = ["tree"]
cat2id = {"tree": 0}  # ID 0 für Detectron2!
categories = [{"id": 0, "name": "tree"}]

print(f"🌳 Training mit 1 Klasse: 'tree' für alle {len(gdf)} Kronen")

# ===========================================
# 🧩 COCO Dataset erstellen
# ===========================================

def create_coco_annotations():
    """Erstelle COCO-kompatible Annotations JSON"""
    coco = {
        "info": {"description": "Single-class tree crown instances"},
        "images": [],
        "annotations": [],
        "categories": categories
    }

    ann_id, img_id = 1, 1
    for tile_file in tqdm(sorted(os.listdir(TILE_DIR)), desc="📷 Processing images"):
        if not tile_file.endswith(".tif"):
            continue

        tile_path = os.path.join(TILE_DIR, tile_file)
        try:
            with rasterio.open(tile_path) as src:
                transform = src.transform
                width, height = src.width, src.height
                tile_bounds = src.bounds
        except Exception as e:
            print(f"⚠️  Fehler beim Lesen {tile_file}: {e}")
            continue

        # Spatial selection
        try:
            subset = gdf.cx[tile_bounds.left:tile_bounds.right,
                            tile_bounds.bottom:tile_bounds.top]
        except Exception:
            subset = gdf

        if len(subset) == 0:
            continue

        # Image registrieren
        coco["images"].append({
            "id": img_id,
            "file_name": os.path.abspath(tile_path),
            "width": width,
            "height": height,
        })

        annotations_for_tile = 0

        # Annotations für dieses Tile - ALLE als Klasse 0 ("tree")
        for _, row in subset.iterrows():
            geom = row.geometry
            if geom.is_empty or geom.area < 0.5:  # Kleine Min-Area
                continue

            # Geometrie zu Pixel-Koordinaten
            inv_transform = ~transform
            seg = []
            try:
                for x, y in geom.exterior.coords:
                    px, py = inv_transform * (x, y)
                    seg.extend([float(px), float(py)])
            except Exception:
                continue

            if len(seg) < 6:  # Mindestens 3 Punkte
                continue

            # Bounding Box berechnen
            xs = seg[0::2]
            ys = seg[1::2]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

            # ALLE Kronen bekommen Category ID 0 ("tree")
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 0,  # Immer 0 für "tree"
                "segmentation": [seg],
                "bbox": bbox,
                "area": geom.area,
                "iscrowd": 0,
            })

            ann_id += 1
            annotations_for_tile += 1

        if annotations_for_tile > 0:
            print(f" {tile_file}: {annotations_for_tile} Kronen")
            img_id += 1
        else:
            coco["images"].pop()  # Entferne Bild ohne Annotations

    return coco

# COCO Dataset erstellen
coco_data = create_coco_annotations()

# Speichern
with open(COCO_OUT_JSON, "w") as f:
    json.dump(coco_data, f, indent=2)

print(f"\n💾 COCO JSON gespeichert: {COCO_OUT_JSON}")
print(f"📊 Statistik:")
print(f"   Images: {len(coco_data['images'])}")
print(f"   Annotations: {len(coco_data['annotations'])}")
print(f"   Categories: {len(coco_data['categories'])}\n")

if len(coco_data['images']) == 0:
    print("❌ Keine Images mit Annotations! Training abgebrochen.")
    exit(1)

# ===========================================
# 🎯 IoU-Berechnung (Hilfsfunktion)
# ===========================================

def compute_iou_batch(pred_masks, gt_masks):
    """
    Berechne durchschnittliche IoU zwischen Pred und GT Masken
    pred_masks: np.array, bool, shape (N, H, W)
    gt_masks: np.array, bool, shape (M, H, W)
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0
    
    ious = []
    for pred_mask in pred_masks:
        max_iou = 0
        for gt_mask in gt_masks:
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            if union > 0:
                iou = float(intersection) / float(union)
                max_iou = max(max_iou, iou)
        ious.append(max_iou)
    
    return float(np.mean(ious)) if ious else 0.0

# ===========================================
# 🎯 ValidationLoss + IoU Hook
# ===========================================

class ValidationLossWithIoU(HookBase):
    """Hook zum Berechnen der Validation Loss + IoU ohne COCOEvaluator (kein OOM)"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        # Nutze TEST-Set als Validation-Loader
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        from detectron2.data import build_detection_train_loader
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        # Nur alle cfg.TEST.EVAL_PERIOD Schritte validieren
        if (self.trainer.iter + 1) % self.cfg.TEST.EVAL_PERIOD != 0:
            return

        data = next(self._loader)
        with torch.no_grad():
            # ===== LOSS =====
            loss_dict = self.trainer.model(data)

            loss_dict_reduced = {
                "val_" + k: v.item()
                for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # ===== IoU =====
            try:
                # Predictions
                outputs = self.trainer.model(data)
                val_iou = 0.0
                
                if isinstance(outputs, list):
                    for idx, output in enumerate(outputs):
                        if "instances" in output:
                            pred_inst = output["instances"]
                            gt_inst = data[idx]["instances"]
                            
                            if len(pred_inst) > 0 and len(gt_inst) > 0:
                                if hasattr(pred_inst, "pred_masks"):
                                    pred_masks = (pred_inst.pred_masks.cpu().numpy() > 0.5).astype(bool)
                                    gt_masks = gt_inst.gt_masks.tensor.cpu().numpy().astype(bool)
                                    batch_iou = compute_iou_batch(pred_masks, gt_masks)
                                    val_iou += batch_iou
                    
                    if len(outputs) > 0:
                        val_iou = val_iou / len(outputs)
                
            except Exception as e:
                print(f"⚠️  IoU-Berechnung Fehler: {e}")
                val_iou = 0.0

            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    total_val_loss=losses_reduced,
                    val_iou=val_iou,
                    **loss_dict_reduced
                )

# ===========================================
# 🎯 Training IoU Hook (leicht)
# ===========================================

class TrainingIoUHook(HookBase):
    """Hook zum Berechnen der Training IoU selten (pro Epoch)"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._train_loader = None
    
    def before_train(self):
        from detectron2.data import build_detection_train_loader
        self._train_loader = iter(build_detection_train_loader(self.cfg))
    
    def after_step(self):
        # Nur alle 500 Iterationen (selten) Training IoU berechnen
        if (self.trainer.iter + 1) % self.cfg.TEST.EVAL_PERIOD != 0:    #% 500 != 0:
            return
        
        try:
            data = next(self._train_loader)
            with torch.no_grad():
                outputs = self.trainer.model(data)
                train_iou = 0.0
                
                if isinstance(outputs, list):
                    for idx, output in enumerate(outputs):
                        if "instances" in output:
                            pred_inst = output["instances"]
                            gt_inst = data[idx]["instances"]
                            
                            if len(pred_inst) > 0 and len(gt_inst) > 0:
                                if hasattr(pred_inst, "pred_masks"):
                                    pred_masks = (pred_inst.pred_masks.cpu().numpy() > 0.5).astype(bool)
                                    gt_masks = gt_inst.gt_masks.tensor.cpu().numpy().astype(bool)
                                    batch_iou = compute_iou_batch(pred_masks, gt_masks)
                                    train_iou += batch_iou
                    
                    if len(outputs) > 0:
                        train_iou = train_iou / len(outputs)
                
                if comm.is_main_process():
                    self.trainer.storage.put_scalar("train_iou", train_iou)
        
        except Exception as e:
            print(f"⚠️  Training IoU Fehler: {e}")

# ===========================================
# 🏋️ Custom Trainer mit ValidationLoss + IoU
# ===========================================

class EnhancedTrainerWithIoU(DefaultTrainer):
    """Trainer mit ValidationLoss + IoU Logging"""

    def build_hooks(self):
        hooks = super().build_hooks()
        if len(self.cfg.DATASETS.TEST) > 0:
            hooks.insert(-1, ValidationLossWithIoU(self.cfg))
            hooks.insert(-1, TrainingIoUHook(self.cfg))
        return hooks

# ===========================================
# 🚀 Detectron2 Training Setup - OPTIMIERT
# ===========================================

# Dataset registrieren
DATASET_NAME = "trees_single_class_with_iou"
register_coco_instances(DATASET_NAME, {}, COCO_OUT_JSON, TILE_DIR)

# Config erstellen
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# GPU Memory Management
torch.cuda.empty_cache()

# Dataset - WICHTIG: TEST-Set für Validation-Hook
cfg.DATASETS.TRAIN = (DATASET_NAME,)
cfg.DATASETS.TEST = (DATASET_NAME,)  # damit ValidationLoss Hook was hat

# Speicher-optimiert
cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH

# Input-Größen für GPU-Management
cfg.INPUT.MIN_SIZE_TRAIN = (600, 700, 800)  # Kleinere Größen für Stabilität
cfg.INPUT.MAX_SIZE_TRAIN = 900              # Reduziert für 8GB GPU
cfg.INPUT.MIN_SIZE_TEST = 600
cfg.INPUT.MAX_SIZE_TEST = 900

# Model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # NUR 1 Klasse!
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ⭐ KRITISCH: Angepasste Parameter für mehr Detections
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000  # Mehr Proposals
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST  = 6000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = 2000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Mehr ROIs pro Image
cfg.TEST.DETECTIONS_PER_IMAGE = 500             # Mehr finale Detections

# Niedrigere NMS-Schwellen für mehr Erkennungen
cfg.MODEL.RPN.NMS_THRESH = 0.6          # Weniger aggressive RPN NMS
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4  # Weniger aggressive finale NMS

# Score-Schwellen anpassen
cfg.MODEL.RPN.SCORE_THRESH_TEST = 0.0  # Niedrigere RPN Score-Schwelle
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Niedrigere finale Score-Schwelle

# Training
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.SOLVER.STEPS = (2000,)  # LR-Reduktion bei 2000
cfg.SOLVER.GAMMA = 0.1      # Starke LR-Reduktion
cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS

# 90 „Pseudo-Epochen" über MAX_ITER=3000
iters_per_epoch = MAX_ITER // N_EPOCHS
cfg.TEST.EVAL_PERIOD = iters_per_epoch
cfg.SOLVER.CHECKPOINT_PERIOD = MAX_ITER + 1

# Output
cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ===========================================
# 🏋️ Training starten...
# ===========================================

print(f"🚀 Starte Enhanced Single-Class Training WITH IoU:")
print(f"   Klassen: 1 ('tree')")
print(f"   Iterationen: {MAX_ITER}")
print(f"   Pseudo-Epochen: {N_EPOCHS} (Eval alle {iters_per_epoch} Iter)")
print(f"   Batch Size: {IMS_PER_BATCH}")
print(f"   Learning Rate: {BASE_LR}")
print(f"   Max Detections: {cfg.TEST.DETECTIONS_PER_IMAGE}")
print(f"   Device: {cfg.MODEL.DEVICE}")
print(f"   📊 IoU wird geloggt (Train: alle 500 Iter, Val: alle {iters_per_epoch} Iter)\n")

try:
    trainer = EnhancedTrainerWithIoU(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("\n✅ Training erfolgreich abgeschlossen!")
    print(f"📍 Modell gespeichert in: {OUTPUT_DIR}\n")

except Exception as e:
    print(f"\n❌ Training Fehler: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

finally:
    torch.cuda.empty_cache()

# ===========================================
# 🔄 JSON → CSV KONVERTIERUNG
# ===========================================

print("🔄 Konvertiere metrics.json zu CSV...\n")

json_path = os.path.join(OUTPUT_DIR, "metrics.json")
csv_output = os.path.join(OUTPUT_DIR, "metrics_detectree2.csv")

try:
    # Lese metrics.json
    rows = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Zeile konnte nicht geparst werden: {e}")
                continue
    
    if not rows:
        print("❌ Keine Daten in metrics.json gefunden!")
        exit(1)
    
    # Konvertiere zu DataFrame
    df = pd.DataFrame(rows)
    
    print(f"✅ metrics.json geladen: {len(df)} Zeilen\n")
    print(f"📋 Verfügbare Spalten:")
    for col in sorted(df.columns):
        non_null_count = df[col].notna().sum()
        print(f"   • {col}: {non_null_count} Werte")
    
    # Stelle sicher, dass kritische Spalten existieren
    required_cols = ['iteration', 'total_loss', 'total_val_loss', 'train_iou', 'val_iou']
    
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️  Spalte '{col}' nicht gefunden, wird mit NaN gefüllt")
            df[col] = float('nan')
    
    # Sortiere nach iteration
    df = df.sort_values('iteration', na_position='last').reset_index(drop=True)
    
    # Speichere als CSV
    df.to_csv(csv_output, index=False)
    
    print(f"\n✅ CSV erstellt: {csv_output}")
    print(f"   Zeilen: {len(df)}")
    print(f"   Spalten: {len(df.columns)}")
    print(f"\n📊 CSV-Header (Loss-Spalten):")
    loss_cols = [col for col in df.columns if 'loss' in col.lower()]
    if loss_cols:
        print(f"   {', '.join(loss_cols[:5])}")
    
    print(f"\n📊 CSV-Header (IoU-Spalten):")
    iou_cols = [col for col in df.columns if 'iou' in col.lower()]
    if iou_cols:
        print(f"   {', '.join(iou_cols)}")
    else:
        print(f"   ⚠️  Keine IoU-Spalten gefunden (Check: wurde Training geloggt?)")
    
except Exception as e:
    print(f"❌ CSV-Konvertierung Fehler: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===========================================
# 🎉 FERTIG
# ===========================================

print("\n╔═══════════════════════════════════════════════════════════════╗")
print("║ ✅ TRAINING + VALIDATION + IoU + CSV ABGESCHLOSSEN!          ║")
print("║                                                               ║")
print("║ 📊 Outputs:                                                   ║")
print(f"║ • metrics.json: {os.path.join(OUTPUT_DIR, 'metrics.json')}")
print(f"║ • metrics_detectree2.csv: {csv_output}")
print("║                                                               ║")
print("║ ✅ Metriken vorhanden:                                        ║")
print("║   • total_loss (Training)                                    ║")
print("║   • total_val_loss (Validation)                              ║")
print("║   • train_iou (Training IoU, alle 500 Iter)                  ║")
print("║   • val_iou (Validation IoU, pro Epoch)                      ║")
print("║   • loss_cls, loss_mask, loss_box_reg, loss_rpn_*            ║")
print("║   • val_loss_cls, val_loss_mask, val_loss_box_reg, ...       ║")
print("║   • lr (Learning Rate)                                       ║")
print("║                                                               ║")
print("║ 🎯 Nächster Schritt:                                          ║")
print("║ Rscript train_compare_COMPLETE.R                             ║")
print("╚═══════════════════════════════════════════════════════════════╝\n")

