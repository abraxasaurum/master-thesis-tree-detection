# ═══════════════════════════════════════════════════════════════════════════
# 📊 PIPE-1 UNIVERSAL – INFERENCE EVALUATION (TCD, Detectree2, DeepTree)
# • Verarbeitet Inference-Outputs aller drei Methoden
# • 1-zu-1-Matching mit Ground Truth (IoU ≥ 0.3)
# • Export: TP-Matches als CSV + GPKG pro Tile, Methode UND GT-Set (train/test)
# • Metriken: TP, FP, FN, Precision, Recall, mean_IoU – GETRENNT nach train/test
# ═══════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(tidyverse)
  library(units)
})

# ═══ 1. KONFIGURATION ═══

gt_path <- "/home/abrax/Desktop/SHP_BEACKUP/ALL_FINAL.shp"  # GT mit 'set'-Spalte (train/test)
output_dir <- "//home/abrax/Desktop/Infer_stat_input/inference_comparison"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Tile-IDs und zugehörige Baumarten
tile_ids <- c(1, 2, 3)
species_names <- c("Pine", "Douglas Fir", "Beech")

# IoU-Schwellenwert für Matching
iou_thr <- 0.30

# Methodenpfade
method_configs <- list(
  TCD = list(
    dir = "/home/abrax/Desktop/Infer_stat_input/tcd/raw",
    pattern = "result_tile%d.sqlite",
    name = "TCD"
  ),
  Detectree2 = list(
    dir = "/home/abrax/Desktop/Infer_stat_input/dt2/raw",
    pattern = "result_tile%d.gpkg",
    name = "Detectree2"
  ),
  DeepTree = list(
    dir = "/home/abrax/Desktop/dp_output",
    pattern = "result_tile%d.gpkg",
    name = "DeepTree"
  )
)

# ═══ 2. GROUND TRUTH LADEN ═══

gt_all <- st_transform(st_read(gt_path, quiet = TRUE), 32633)

# Validierungen
if (!"tile" %in% names(gt_all)) stop("❌ GT fehlt 'tile'-Spalte!")
if (!"set" %in% names(gt_all)) {
  cat("⚠️  'set'-Spalte nicht gefunden – setze standardmäßig auf 'train'\n")
  gt_all$set <- "train"
}

gt_tiles <- split(gt_all, gt_all$tile)
cat("✓ Ground Truth geladen und nach Tiles aufgeteilt\n")
cat(sprintf("  Train-Objekte: %d\n", sum(gt_all$set == "train", na.rm = TRUE)))
cat(sprintf("  Test-Objekte: %d\n", sum(gt_all$set == "test", na.rm = TRUE)))

# ═══ 3. HILFSFUNKTIONEN ═══

# Inference-Output einlesen (GPKG oder SQLite)
read_inference <- function(file_path) {
  if (!file.exists(file_path)) return(NULL)
  lyr <- st_layers(file_path)$name[1]
  st_transform(st_read(file_path, layer = lyr, quiet = TRUE), 32633)
}

# IoU-Berechnung
iou <- function(a, b) {
  inter <- suppressWarnings(st_intersection(a, b))
  if (st_is_empty(inter)) return(0)
  Ai <- st_area(inter)
  Au <- st_area(a) + st_area(b) - Ai
  as.numeric(Ai / Au)
}

# 1:1 Matching (höchster IoU pro GT-Polygon)
match_tile <- function(gt, det, thr = iou_thr) {
  cand <- st_intersects(gt, det)
  res <- vector("list", nrow(gt))
  for (i in seq_along(cand)) {
    best <- 0
    bestj <- NA
    for (j in cand[[i]]) {
      val <- iou(gt[i, ], det[j, ])
      if (val > best) {
        best <- val
        bestj <- j
      }
    }
    if (!is.na(bestj) && best >= thr)
      res[[i]] <- data.frame(gt_id = i, det_id = bestj, iou = best)
  }
  bind_rows(res)
}

# ═══ 4. HAUPTSCHLEIFE: ALLE METHODEN × ALLE TILES × TRAIN/TEST ═══

all_results <- list()
all_tp_combined <- list()

for (method_key in names(method_configs)) {
  method <- method_configs[[method_key]]
  cat(sprintf("\n════════════════════════════════════════\n"))
  cat(sprintf("🔬 METHODE: %s\n", method$name))
  cat(sprintf("════════════════════════════════════════\n"))
  
  for (tid in tile_ids) {
    cat(sprintf("\n🔍 Tile %d (%s) – Verarbeitung...\n", tid, species_names[tid]))
    
    # Inference-Datei laden
    inf_file <- file.path(method$dir, sprintf(method$pattern, tid))
    det <- read_inference(inf_file)
    
    if (is.null(det)) {
      cat(sprintf("⚠️  Datei nicht gefunden: %s\n", inf_file))
      next
    }
    
    # Ground Truth für dieses Tile (alle polygone zunächst)
    gt_tile <- gt_tiles[[as.character(tid)]]
    
    # Fallback für fehlende 'fid'
    if (!"fid" %in% names(gt_tile)) gt_tile$fid <- seq_len(nrow(gt_tile))
    if (!"fid" %in% names(det)) det$fid <- seq_len(nrow(det))
    
    # ──────── TRAIN/TEST SUBSET LOOP ────────
    for (subset_name in c("train", "test")) {
      # Filtere GT nach set
      gt <- gt_tile %>% filter(set == subset_name)
      
      if (nrow(gt) == 0) {
        cat(sprintf("  ℹ️  [%s] Keine Polygone in diesem Set\n", subset_name))
        next
      }
      
      # Matching durchführen
      matches <- match_tile(gt, det)
      
      # ──────── 4.1 | TP-Export als GPKG (mit Geometrie) ────────
      if (nrow(matches) > 0) {
        tp_geo <- gt[matches$gt_id, ]
        tp_geo$tp_iou <- matches$iou
        tp_geo$det_id <- matches$det_id
        tp_geo$tile <- tid
        tp_geo$method <- method$name
        tp_geo$gt_set <- subset_name
        tp_geo <- tp_geo %>% select(-any_of("fid"))
        
        gpkg_out <- file.path(
          output_dir,
          sprintf("pipe1_TP_%s_%s_tile%d.gpkg", method$name, subset_name, tid)
        )
        if (file.exists(gpkg_out)) file.remove(gpkg_out)
        st_write(tp_geo, gpkg_out, delete_layer = TRUE, quiet = TRUE)
        cat(sprintf("  ✓ GPKG exportiert: %s\n", basename(gpkg_out)))
      }
      
      # ──────── 4.2 | TP-Export als CSV (nur Attribute) ────────
      if (nrow(matches) > 0) {
        matches_csv <- matches %>%
          mutate(
            gt_fid = gt$fid[gt_id],
            det_fid = det$fid[det_id],
            tile = tid,
            method = method$name,
            gt_set = subset_name,
            detection_file = basename(inf_file)
          )
        
        csv_out <- file.path(
          output_dir,
          sprintf("pipe1_TP_%s_%s_tile%d.csv", method$name, subset_name, tid)
        )
        write_csv(matches_csv, csv_out)
        cat(sprintf("  ✓ CSV exportiert: %s\n", basename(csv_out)))
        
        # Für spätere Gesamtauswertung
        all_tp_combined[[length(all_tp_combined) + 1]] <- matches_csv
      }
      
      # ──────── 4.3 | Metriken berechnen ────────
      TP <- nrow(matches)
      FP <- nrow(det) - length(unique(matches$det_id))
      FN <- nrow(gt) - TP
      
      all_results[[length(all_results) + 1]] <- data.frame(
        method = method$name,
        tile = tid,
        species = species_names[tid],
        gt_set = subset_name,
        gt_total = nrow(gt),
        det_total = nrow(det),
        TP = TP,
        FP = FP,
        FN = FN,
        Recall = round(TP / (TP + FN), 3),
        Precision = round(TP / (TP + FP), 3),
        F1 = round(2 * TP / (2 * TP + FP + FN), 3),
        mean_IoU = if (TP > 0) round(mean(matches$iou), 3) else NA
      )
      
      cat(sprintf("  [%s] TP=%d, FP=%d, FN=%d, Recall=%.3f, Precision=%.3f\n",
                  subset_name, TP, FP, FN,
                  TP / (TP + FN),
                  TP / (TP + FP)))
    }
  }
}

# ═══ 5. GESAMTAUSWERTUNG ═══

results_tbl <- bind_rows(all_results)
write_csv(results_tbl, file.path(output_dir, "pipe1_summary_all_methods.csv"))

cat("\n\n════════════════════════════════════════\n")
cat("📋 GESAMTAUSWERTUNG – ALLE METHODEN\n")
cat("════════════════════════════════════════\n")
print(results_tbl)

# ═══ 6. TP-KOMBINIERTE TABELLE ═══

if (length(all_tp_combined) > 0) {
  tp_all <- bind_rows(all_tp_combined)
  write_csv(tp_all, file.path(output_dir, "pipe1_TP_ALL_COMBINED.csv"))
  cat(sprintf("\n✓ Kombinierte TP-Tabelle exportiert: %d Matches\n", nrow(tp_all)))
}

# ═══ 7. ZUSÄTZLICHE STATISTIK: TRAIN vs TEST VERGLEICH ═══

cat("\n════════════════════════════════════════\n")
cat("📊 VERGLEICH: TRAIN vs TEST\n")
cat("════════════════════════════════════════\n")

stats_by_set <- results_tbl %>%
  group_by(method, gt_set) %>%
  summarise(
    n_tiles = n(),
    total_gt = sum(gt_total, na.rm = TRUE),
    total_det = sum(det_total, na.rm = TRUE),
    total_TP = sum(TP, na.rm = TRUE),
    total_FP = sum(FP, na.rm = TRUE),
    total_FN = sum(FN, na.rm = TRUE),
    mean_Recall = round(mean(Recall, na.rm = TRUE), 3),
    mean_Precision = round(mean(Precision, na.rm = TRUE), 3),
    mean_F1 = round(mean(F1, na.rm = TRUE), 3),
    mean_IoU = round(mean(mean_IoU, na.rm = TRUE), 3),
    .groups = "drop"
  )

print(stats_by_set)
write_csv(stats_by_set, file.path(output_dir, "pipe1_summary_by_set.csv"))

# ═══ 8. LEGENDE ═══

cat("\n📘 Legende\n")
cat(" TP = True Positives (korrekt detektiert)\n")
cat(" FP = False Positives (Übersegmentierung/nicht gematchte Detektionen)\n")
cat(" FN = False Negatives (nicht detektiert)\n")
cat(" Recall = TP / (TP + FN) – Empfindlichkeit\n")
cat(" Precision = TP / (TP + FP) – Genauigkeit\n")
cat(" F1 = 2*TP / (2*TP + FP + FN) – harmonischer Mittelwert\n")
cat(" mean_IoU = Ø IoU der True Positives\n")
cat(" gt_set = 'train' (85%) oder 'test' (15% Holdout)\n")

cat("\n✅ PIPE-1 UNIVERSAL abgeschlossen!\n")
cat(sprintf("📂 Ergebnisse: %s\n", output_dir))
cat("\n📄 Erzeugte Dateien:\n")
cat(" • pipe1_summary_all_methods.csv (Train + Test gemischt)\n")
cat(" • pipe1_summary_by_set.csv (Train vs Test aggregiert)\n")
cat(" • pipe1_TP_<Methode>_<Set>_tile<id>.gpkg (je TP-Polygon mit Geometrie)\n")
cat(" • pipe1_TP_<Methode>_<Set>_tile<id>.csv (je TP-Match-Info)\n")
cat(" • pipe1_TP_ALL_COMBINED.csv (alle TP-Matches vereinigt)\n")
