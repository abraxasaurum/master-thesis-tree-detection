#!/usr/bin/env python3
"""
detectree2_infer_instances_ENHANCED_FIXED.py
Optimierte Inferenz mit automatischer Bildgrößen-Begrenzung
"""

import os
import numpy as np
import cv2
import torch
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# ===========================================
# 🔧 Konfiguration
# ===========================================
BASE_DIR = "/home/abrax/detectree2/training"

# ANPASSEN: Pfad zu deinem trainierten Modell
MODEL_PATH = os.path.join(BASE_DIR, "outputs_single_class_with_final/model_final.pth")

# Input und Output
INPUT_IMAGE = os.path.join(BASE_DIR, "images", "martel_rgb.tif")
OUTPUT_GPKG = os.path.join(BASE_DIR, "result_martel_rgb.gpkg") ##
OUTPUT_VISUAL = os.path.join(BASE_DIR, "detection_visualization.jpg") ##

# Inference-Parameter
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
MAX_DETECTIONS = 1000

# KRITISCH: Maximale Bildgröße begrenzen
MAX_IMAGE_SIZE = 1000  # Max. 4000 Pixel pro Seite für RAM-Management

# Polygon-Verarbeitung
MIN_AREA = 3.0
SIMPLIFY_TOLERANCE = 0.3

print(f"🎯 Enhanced Tree Crown Detection (RAM-optimiert)")
print(f"📂 Modell: {MODEL_PATH}")
print(f"🖼️ Input: {INPUT_IMAGE}")
print(f"📊 Score Threshold: {SCORE_THRESHOLD}")
print(f"🔍 Max Detections: {MAX_DETECTIONS}")
print(f"📐 Max Image Size: {MAX_IMAGE_SIZE}px")

# ===========================================
# 🤖 Modell laden
# ===========================================
def setup_predictor():
    """Predictor für optimierte Kronenerkennung einrichten"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Single-Class Setup
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reduzierte Input-Größen für RAM-Management
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = MAX_IMAGE_SIZE
    
    # Erweiterte Detection-Parameter
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.NMS_THRESH = 0.6
    cfg.MODEL.RPN.SCORE_THRESH_TEST = 0.0
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = NMS_THRESHOLD
    cfg.TEST.DETECTIONS_PER_IMAGE = MAX_DETECTIONS
    
    # Metadata für Single-Class
    MetadataCatalog.get("tree_inference").set(thing_classes=["tree"])
    cfg.DATASETS.TEST = ("tree_inference",)
    
    return DefaultPredictor(cfg)

print("🔧 Lade Modell...")
predictor = setup_predictor()
print(f"✅ Modell geladen auf: {predictor.cfg.MODEL.DEVICE}")

# ===========================================
# 📍 Bild laden und ggf. verkleinern
# ===========================================
print("📖 Lade und verarbeite Eingangsbild...")
with rasterio.open(INPUT_IMAGE) as src:
    original_transform = src.transform
    original_crs = src.crs
    original_width = src.width
    original_height = src.height
    
    print(f"📐 Original-Bildgröße: {original_width} × {original_height}")
    
    # Berechne Skalierungsfaktor
    max_dim = max(original_width, original_height)
    if max_dim > MAX_IMAGE_SIZE:
        scale_factor = MAX_IMAGE_SIZE / max_dim
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        print(f"📉 Skaliere auf: {new_width} × {new_height} (Faktor: {scale_factor:.3f})")
        
        # Resample mit rasterio
        from rasterio.enums import Resampling
        image_rgb = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        ).transpose(1, 2, 0)
        
        # Angepassten Transform berechnen
        transform = original_transform * original_transform.scale(
            original_width / new_width,
            original_height / new_height
        )
        
    else:
        # Keine Skalierung nötig
        scale_factor = 1.0
        new_width, new_height = original_width, original_height
        image_rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        transform = original_transform
        print(f"✅ Bildgröße OK, keine Skalierung nötig")

print(f"📐 Finale Bildgröße für Inferenz: {new_width} × {new_height}")
print(f"🗺️ CRS: {original_crs}")

# Konvertiere zu BGR für OpenCV/Detectron2
image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# ===========================================
# 🔍 Inferenz durchführen
# ===========================================
print("🚀 Starte Kronenerkennung...")
with torch.no_grad():
    torch.cuda.empty_cache()
    outputs = predictor(image_bgr)

instances = outputs["instances"].to("cpu")
num_detections = len(instances)
print(f"🎯 {num_detections} Kronen erkannt")

if num_detections == 0:
    print("❌ Keine Kronen erkannt!")
    exit(1)

# ===========================================
# 📊 Detections analysieren
# ===========================================
scores = instances.scores.numpy()
masks = instances.pred_masks.numpy()
boxes = instances.pred_boxes.tensor.numpy()

print(f"📈 Score-Statistiken:")
print(f"   Min: {scores.min():.3f}")
print(f"   Max: {scores.max():.3f}")
print(f"   Median: {np.median(scores):.3f}")
print(f"   Über {SCORE_THRESHOLD}: {(scores >= SCORE_THRESHOLD).sum()}")

# ===========================================
# 🗺️ Masken zu Polygonen konvertieren (mit Skalierung)
# ===========================================
def mask_to_polygon(mask, transform, scale_factor, min_area=MIN_AREA):
    """Konvertiert Maske zu georeferenziertem Polygon"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        if len(contour) < 4:
            continue
            
        # Pixelkoordinaten zu Weltkoordinaten (mit Rückskalierung!)
        world_coords = []
        for point in contour.reshape(-1, 2):
            # Rückskalierung zu Original-Pixelkoordinaten
            x_pixel = float(point[0]) / scale_factor
            y_pixel = float(point[1]) / scale_factor
            
            # Weltkoordinaten mit Original-Transform
            x_world, y_world = original_transform * (x_pixel, y_pixel)
            world_coords.append((x_world, y_world))
        
        if len(world_coords) >= 3:
            try:
                poly = Polygon(world_coords)
                poly = make_valid(poly)
                
                if poly.is_valid and poly.area >= min_area:
                    if SIMPLIFY_TOLERANCE > 0:
                        poly = poly.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
                    polygons.append(poly)
            except Exception as e:
                continue
    
    return polygons

print("🔄 Konvertiere Masken zu Polygonen...")
all_polygons = []
all_scores = []
all_areas = []

for i in range(num_detections):
    mask = masks[i]
    score = scores[i]
    
    polygons = mask_to_polygon(mask, transform, scale_factor)
    
    for poly in polygons:
        all_polygons.append(poly)
        all_scores.append(score)
        all_areas.append(poly.area)

print(f"✅ {len(all_polygons)} gültige Polygone erstellt")

if len(all_polygons) == 0:
    print("❌ Keine gültigen Polygone!")
    exit(1)

# ===========================================
# 📊 Polygon-Statistiken
# ===========================================
areas = np.array(all_areas)
scores_array = np.array(all_scores)

print(f"📐 Flächen-Statistiken (m²):")
print(f"   Min: {areas.min():.2f}")
print(f"   Max: {areas.max():.2f}")
print(f"   Median: {np.median(areas):.2f}")
print(f"   Gesamt: {areas.sum():.2f}")

# ===========================================
# 💾 GeoPackage speichern
# ===========================================
print("💾 Speichere Ergebnisse...")

gdf = gpd.GeoDataFrame({
    'geometry': all_polygons,
    'confidence': all_scores,
    'area_m2': all_areas,
    'tree_id': range(1, len(all_polygons) + 1),
    'detection_class': ['tree'] * len(all_polygons),
    'scale_factor': [scale_factor] * len(all_polygons)
}, crs=original_crs)

# Nach Confidence sortieren
gdf = gdf.sort_values('confidence', ascending=False).reset_index(drop=True)

# Speichern
gdf.to_file(OUTPUT_GPKG, driver='GPKG')
print(f"✅ {len(gdf)} Kronen gespeichert: {OUTPUT_GPKG}")

# ===========================================
# 🎨 Visualisierung erstellen (auf skaliertem Bild)
# ===========================================
print("🎨 Erstelle Visualisierung...")

v = Visualizer(
    image_rgb[:, :, ::-1],
    MetadataCatalog.get("tree_inference"),
    scale=0.8,
    instance_mode=ColorMode.IMAGE
)

vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
vis_image = vis.get_image()[:, :, ::-1]

cv2.imwrite(OUTPUT_VISUAL, vis_image)
print(f"📸 Visualisierung gespeichert: {OUTPUT_VISUAL}")

# ===========================================
# 📋 Zusammenfassung
# ===========================================
print("\n🎯 === ERGEBNISSE ===")
print(f"📊 Erkannte Kronen: {len(gdf)}")
print(f"📈 Durchschnittliche Confidence: {scores_array.mean():.3f}")
print(f"📐 Durchschnittliche Fläche: {areas.mean():.2f} m²")
print(f"🌳 Gesamte Kronenfläche: {areas.sum():.2f} m²")
print(f"📏 Skalierungsfaktor: {scale_factor:.3f}")
print(f"💾 Output: {OUTPUT_GPKG}")
print(f"🖼️ Visualisierung: {OUTPUT_VISUAL}")

high_conf = gdf[gdf['confidence'] >= 0.7]
print(f"⭐ Hohe Confidence (≥0.7): {len(high_conf)} Kronen")

print("\n✅ Inferenz erfolgreich abgeschlossen!")
torch.cuda.empty_cache()

