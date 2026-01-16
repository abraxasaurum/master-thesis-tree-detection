#!/usr/bin/env Rscript

# ═══════════════════════════════════════════════════════════════
# 📊 PIPE-1 MARTEL – POLYGON-TO-POLYGON MATCHING (TCD, Detectree2)
# • Ground Truth: GT_martel.gpkg (Polygone mit fid + Martel-Attributen)
# • Detections: TCD- und Detectree2-Polygone (raw)
# • 1-zu-1-Matching mit IoU ≥ 0.3
# • Output je Methode: TP-Detektionen/GT als GPKG + Match-CSV + Summary
# ═══════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(readr)
})

# === 1. KONFIGURATION ==========================================

gt_path <- "/home/abrax/Desktop/Martelo/GT_martel.gpkg"

method_configs <- list(
  TCD = list(
    file  = "/home/abrax/Desktop/Martelo/result_infer_martel_rgb_tcd.sqlite",
    layer = NULL,
    name  = "TCD"
  ),
  Detectree2 = list(
    file  = "/home/abrax/Desktop/Martelo/result_infer_martel_rgb_dt2_v1.gpkg",
    layer = NULL,
    name  = "Detectree2"
  )
)

iou_thr    <- 0.30
output_dir <- "/home/abrax/Desktop/Martelo/infer_compare_martel"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# === 2. GT LADEN UND NORMALISIEREN =============================

gt_all <- st_read(gt_path, quiet = TRUE)

if (is.na(st_crs(gt_all))) stop("GT hat kein CRS – bitte in QGIS setzen.")

if (!"fid" %in% names(gt_all)) {
  gt_all$fid <- seq_len(nrow(gt_all))
  warning("GT hatte keine Spalte 'fid' – wurde neu erzeugt.")
}

gt_all <- st_make_valid(gt_all)
gt_all <- st_cast(gt_all, "MULTIPOLYGON", warn = FALSE)

cat("✓ Ground Truth geladen:", nrow(gt_all), "Polygone\n")

# === 3. FUNKTIONEN =============================================

read_detection <- function(file_path, layer_name = NULL, target_crs = st_crs(gt_all)) {
  if (!file.exists(file_path)) return(NULL)
  if (is.null(layer_name)) {
    lyr <- st_layers(file_path)$name[1]
  } else {
    lyr <- layer_name
  }
  det <- st_read(file_path, layer = lyr, quiet = TRUE)
  
  if (is.na(st_crs(det))) {
    st_crs(det) <- target_crs
  } else {
    det <- st_transform(det, target_crs)
  }
  
  det <- st_make_valid(det)
  
  # GeometryCollections und andere Exoten rauswerfen
  gtype <- as.character(st_geometry_type(det))
  det <- det[gtype %in% c("POLYGON", "MULTIPOLYGON"), ]
  
  # Auf MULTIPOLYGON casten
  det <- st_cast(det, "MULTIPOLYGON", warn = FALSE)
  
  # Leere / flächenlose Geometrien entfernen
  det[["__area"]] <- st_area(det)
  det <- det[!st_is_empty(det) & as.numeric(det[["__area"]]) > 0, ]
  det[["__area"]] <- NULL
  
  det
}

iou_geom <- function(geom_a, geom_b) {
  inter <- suppressWarnings(st_intersection(geom_a, geom_b))
  if (length(inter) == 0 || all(st_is_empty(inter))) return(0)
  Ai <- st_area(inter)
  Ai <- sum(Ai)
  Au <- st_area(geom_a) + st_area(geom_b) - Ai
  as.numeric(Ai / Au)
}

match_polygons <- function(gt, det, thr = iou_thr) {
  g_geom <- st_geometry(gt)
  d_geom <- st_geometry(det)
  cand   <- st_intersects(g_geom, d_geom)
  res    <- vector("list", length = length(g_geom))
  
  for (i in seq_along(cand)) {
    if (length(cand[[i]]) == 0) next
    best  <- 0
    bestj <- NA_integer_
    for (j in cand[[i]]) {
      val <- iou_geom(g_geom[i], d_geom[j])
      if (val > best) {
        best  <- val
        bestj <- j
      }
    }
    if (!is.na(bestj) && best >= thr) {
      res[[i]] <- data.frame(
        gt_id  = i,
        det_id = bestj,
        iou    = best
      )
    }
  }
  bind_rows(res)
}

# === 4. HAUPTSCHLEIFE ==========================================

all_results <- list()
all_tp_combined <- list()

for (method_key in names(method_configs)) {
  method <- method_configs[[method_key]]
  cat("\n════════════════════════════════════════\n")
  cat(sprintf("🔬 METHODE: %s\n", method$name))
  cat(sprintf("Datei: %s\n", method$file))
  cat("════════════════════════════════════════\n")
  
  det <- read_detection(method$file, method$layer, st_crs(gt_all))
  if (is.null(det)) {
    cat("⚠️  Detection-Datei nicht gefunden, überspringe.\n")
    next
  }
  
  # Interne ID-Spalten aus dem Input entfernen, um Konflikte beim Schreiben zu vermeiden
  id_in <- intersect(names(det), c("ogc_fid", "FID"))
  if (length(id_in) > 0) {
    det <- det %>% select(-all_of(id_in))
  }
  
  if (!"fid" %in% names(det)) {
    det$fid <- seq_len(nrow(det))
    warning("Detection-Layer hatte keine Spalte 'fid' – wurde neu erzeugt.")
  }
  
  cat("✓ Detections nach Cleaning:", nrow(det), "\n")
  
  cat("🔁 Starte IoU-Matching GT ↔ Detections ...\n")
  matches <- match_polygons(gt_all, det, thr = iou_thr)
  TP <- nrow(matches)
  cat(sprintf("✓ Matches gefunden: TP=%d\n", TP))
  
  FP <- nrow(det) - length(unique(matches$det_id))
  FN <- nrow(gt_all) - TP
  
  Recall    <- if ((TP + FN) > 0) TP / (TP + FN) else NA
  Precision <- if ((TP + FP) > 0) TP / (TP + FP) else NA
  F1        <- if (!is.na(Recall) && !is.na(Precision) && (Recall + Precision) > 0) {
    2 * Recall * Precision / (Recall + Precision)
  } else NA
  mean_IoU  <- if (TP > 0) mean(matches$iou) else NA
  
  all_results[[length(all_results) + 1]] <- data.frame(
    method    = method$name,
    gt_total  = nrow(gt_all),
    det_total = nrow(det),
    TP        = TP,
    FP        = FP,
    FN        = FN,
    Recall    = round(Recall, 3),
    Precision = round(Precision, 3),
    F1        = round(F1, 3),
    mean_IoU  = round(mean_IoU, 3)
  )
  
  cat(sprintf("  TP=%d, FP=%d, FN=%d, Recall=%.3f, Precision=%.3f, F1=%.3f, mean_IoU=%.3f\n",
              TP, FP, FN,
              Recall, Precision, F1, mean_IoU))
  
  if (TP > 0) {
    tp_gt  <- gt_all[matches$gt_id, ]
    tp_det <- det[matches$det_id, ]
    
    tp_det$gt_fid <- tp_gt$fid
    tp_det$gt_iou <- matches$iou
    
    # vor dem Schreiben eigene ID-Spalten entfernen
    id_cols <- intersect(names(tp_det), c("fid", "ogc_fid", "FID"))
    if (length(id_cols) > 0) {
      tp_det <- tp_det %>% select(-all_of(id_cols))
    }
    
    gpkg_out_det <- file.path(
      output_dir,
      sprintf("martel_TP_%s_detections.gpkg", method$name)
    )
    if (file.exists(gpkg_out_det)) file.remove(gpkg_out_det)
    st_write(tp_det, gpkg_out_det, delete_layer = TRUE, quiet = TRUE)
    cat(sprintf("  ✓ TP-Detections exportiert: %s\n", basename(gpkg_out_det)))
    
    gpkg_out_gt <- file.path(
      output_dir,
      sprintf("martel_TP_%s_GT.gpkg", method$name)
    )
    if (file.exists(gpkg_out_gt)) file.remove(gpkg_out_gt)
    st_write(tp_gt, gpkg_out_gt, delete_layer = TRUE, quiet = TRUE)
    cat(sprintf("  ✓ Zugehörige GT-Polygone exportiert: %s\n", basename(gpkg_out_gt)))
    
    matches_csv <- matches %>%
      mutate(
        gt_fid  = gt_all$fid[gt_id],
        det_fid = det$fid[det_id],
        method  = method$name
      )
    csv_out <- file.path(
      output_dir,
      sprintf("martel_TP_%s_matches.csv", method$name)
    )
    write_csv(matches_csv, csv_out)
    cat(sprintf("  ✓ TP-Match-Tabelle exportiert: %s\n", basename(csv_out)))
    
    all_tp_combined[[length(all_tp_combined) + 1]] <- matches_csv
  } else {
    cat("  ⚠️ Keine True Positives für diese Methode.\n")
  }
}

# === 5. GESAMTAUSWERTUNG ======================================

if (length(all_results) > 0) {
  results_tbl <- bind_rows(all_results)
  write_csv(results_tbl, file.path(output_dir, "martel_summary_all_methods.csv"))
  
  cat("\n\n════════════════════════════════════════\n")
  cat("📋 GESAMTAUSWERTUNG – ALLE METHODEN\n")
  cat("════════════════════════════════════════\n")
  print(results_tbl)
}

# === 6. KOMBINIERTE TP-TABELLE =================================

if (length(all_tp_combined) > 0) {
  tp_all <- bind_rows(all_tp_combined)
  write_csv(tp_all, file.path(output_dir, "martel_TP_ALL_COMBINED.csv"))
  cat(sprintf("\n✓ Kombinierte TP-Tabelle exportiert: %d Matches\n", nrow(tp_all)))
}

# === 7. LEGENDE ================================================

cat("\n📘 Legende\n")
cat(" TP = True Positives (Detection ↔ GT mit IoU ≥ Schwelle)\n")
cat(" FP = False Positives (Detections ohne GT-Match im GT-Subset)\n")
cat(" FN = False Negatives (GT-Polygone im GT-Subset ohne Detection)\n")
cat(" Recall = TP / (TP + FN)\n")
cat(" Precision = TP / (TP + FP)\n")
cat(" F1 = 2 * Precision * Recall / (Precision + Recall)\n")
cat(" mean_IoU = Mittelwert der IoUs aller TP\n")

cat("\n✅ PIPE-1 MARTEL (Polygon-zu-Polygon) abgeschlossen!\n")
cat(sprintf("📂 Ergebnisse: %s\n", output_dir))
cat("\n📄 Erzeugte Dateien (pro Methode):\n")
cat(" • martel_TP_<Methode>_detections.gpkg (TP-Detektionen mit gt_fid, gt_iou)\n")
cat(" • martel_TP_<Methode>_GT.gpkg (zugehörige GT-Polygone)\n")
cat(" • martel_TP_<Methode>_matches.csv (ID-/IoU-Tabelle)\n")
cat(" • martel_TP_ALL_COMBINED.csv (alle Methoden kombiniert)\n")
cat(" • martel_summary_all_methods.csv (Metriken pro Methode)\n")
