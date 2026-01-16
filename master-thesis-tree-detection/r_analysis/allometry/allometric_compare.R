#!/usr/bin/env Rscript

# ----------------------------------------------------------
# PIPE 2 – MultiMethod Statistik mit PIPE1-GT-Link
# ----------------------------------------------------------

library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(purrr)
library(stringr)

# ----------------------------------------------------------
# 1. Pfade und Dateien
# ----------------------------------------------------------

# Ground Truth (TreeVis)
gt_file <- "/home/abrax/Desktop/Infer_stat_input/ALL_DATA_merged_final.csv"

# Ordner mit PIPE1-TP-Tabellen
pipe1_dir <- "/home/abrax/Desktop/Infer_stat_input/inference_comparison"

# Ordner mit den MultiMethod-Resultaten
mm_dir <- "/home/abrax/Desktop/Bitz_singular/output/multimethod"

tiles <- c("tile1", "tile2", "tile3")
methods_use <- c("TCD", "Detectree2")  # DeepTree raus

mm_path <- function(tile, method) {
  file.path(mm_dir, tile, tolower(method), sprintf("%s_%s_results.csv", tile, tolower(method)))
}

# ----------------------------------------------------------
# 2. Ground Truth laden
# ----------------------------------------------------------

gt <- read_csv(gt_file, show_col_types = FALSE)

gt <- gt %>%
  mutate(
    fid        = as.numeric(fid),           # GT-FID
    tr_hgth_m  = as.numeric(tr_hgth),
    dbh_m      = as.numeric(DBH),
    cr_area_m2 = as.numeric(cr_ar_m2),
    species_gt = as.character(ba_text),
    zustand_gt = as.character(zustand),
    tile       = as.numeric(tile)
  ) %>%
  filter(!is.na(fid))

cat("GT – Anzahl Bäume:", nrow(gt), "\n")

# ----------------------------------------------------------
# 3. PIPE1-TP-Links (det_fid -> gt_fid) laden
# ----------------------------------------------------------

# Hilfsfunktion: Pfad für PIPE1-TP-CSV
pipe1_tp_path <- function(method, tile) {
  # method: "TCD" oder "Detectree2"
  # tile:   "tile1".."tile3"
  tile_idx <- as.numeric(sub("tile", "", tile))
  if (method == "TCD") {
    sprintf("%s/pipe1_TP_TCD_train_tile%d.csv", pipe1_dir, tile_idx)
  } else if (method == "Detectree2") {
    sprintf("%s/pipe1_TP_Detectree2_train_tile%d.csv", pipe1_dir, tile_idx)
  } else {
    NA_character_
  }
}

read_pipe1_tp <- function(method, tile) {
  f <- pipe1_tp_path(method, tile)
  if (!file.exists(f)) {
    warning("PIPE1-TP-Datei fehlt: ", f)
    return(NULL)
  }
  read_csv(f, show_col_types = FALSE) %>%
    transmute(
      method = method,
      tile   = as.numeric(sub("tile", "", tile)),
      det_fid = as.numeric(det_fid),
      gt_fid  = as.numeric(gt_fid)
    )
}

pipe1_links <- map_dfr(tiles, function(tile) {
  map_dfr(methods_use, ~ read_pipe1_tp(.x, tile))
})

cat("PIPE1-Links – Zeilen:", nrow(pipe1_links), "\n")

# ----------------------------------------------------------
# 4. MultiMethod-PIPE2-Resultate laden + mit PIPE1-Links verbinden
# ----------------------------------------------------------

read_mm_results <- function(tile, method) {
  f <- mm_path(tile, method)
  if (!file.exists(f)) {
    warning("PIPE2-Result-Datei fehlt: ", f)
    return(NULL)
  }
  
  df <- read_csv(f, show_col_types = FALSE)
  
  df %>%
    mutate(
      tile         = as.numeric(sub("tile", "", tile)),   # 1/2/3
      method       = method,
      fid_det      = as.numeric(fid_det),                     # Detektions-FID
      baumart_pred = as.character(baumart_pred),
      zustand_pred = as.character(zustand_pred),
      chm_mean_m   = as.numeric(chm_mean),
      crown_m2     = as.numeric(polygon_area_m2),
      dbh_pred_cm  = as.numeric(bhd_height_crown_cm),
      dbh_pred_m   = dbh_pred_cm / 100
    )
}

mm_all <- map_dfr(tiles, function(tile) {
  map_dfr(methods_use, ~ read_mm_results(tile, .x))
})

cat("PIPE2-Resultate – Zeilen:", nrow(mm_all), "\n")

# 4a. PIPE2 + PIPE1-Links verbinden -> GT-FID
mm_with_gtfid <- mm_all %>%
  left_join(
    pipe1_links,
    by = c("method", "tile", "fid_det" = "det_fid")
  )
table(mm_with_gtfid$method, is.na(mm_with_gtfid$gt_fid))

# ----------------------------------------------------------
# 5. PIPE2+PIPE1 mit GT verbinden
# ----------------------------------------------------------

data_merged <- mm_with_gtfid %>%
  left_join(
    gt %>%
      select(fid, tile, species_gt, tr_hgth_m, dbh_m, cr_area_m2, zustand_gt),
    by = c("gt_fid" = "fid", "tile")
  )

cat("Nach GT-Join – Zeilen:", nrow(data_merged), "\n")
cat("Mit GT-Höhe:", sum(!is.na(data_merged$tr_hgth_m)), "\n")

# ----------------------------------------------------------
# 6. Diagnose: fehlende GT-Link oder GT-Einträge
# ----------------------------------------------------------

missing_links <- data_merged %>%
  filter(is.na(gt_fid))

missing_gt <- data_merged %>%
  filter(!is.na(gt_fid) & (is.na(tr_hgth_m) | is.na(dbh_m) | is.na(cr_area_m2)))

out_dir <- file.path(mm_dir, "pipe2_statistics")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_csv(missing_links, file.path(out_dir, "pipe2_missing_pipe1_link.csv"))
write_csv(missing_gt,    file.path(out_dir, "pipe2_missing_gt_after_link.csv"))

# Für die eigentliche Auswertung nur Zeilen mit vollständiger GT-Info nehmen
data_eval <- data_merged %>%
  filter(!is.na(tr_hgth_m), !is.na(dbh_m), !is.na(cr_area_m2))

cat("Für Auswertung verwendete Bäume:", nrow(data_eval), "\n")

# ----------------------------------------------------------
# 7. Klassifikations-Performance (Art & Zustand)
# ----------------------------------------------------------

species_stats <- data_eval %>%
  mutate(
    species_pred = baumart_pred,
    species_gt   = species_gt,
    correct_species = species_pred == species_gt
  ) %>%
  group_by(method, species_gt) %>%
  summarise(
    n           = n(),
    accuracy    = mean(correct_species),
    n_correct   = sum(correct_species),
    n_incorrect = n - n_correct,
    .groups = "drop"
  )

zustand_stats <- data_eval %>%
  mutate(
    cond_pred = zustand_pred,
    cond_gt   = zustand_gt,
    correct_cond = cond_pred == cond_gt
  ) %>%
  group_by(method, cond_gt) %>%
  summarise(
    n           = n(),
    accuracy    = mean(correct_cond),
    n_correct   = sum(correct_cond),
    n_incorrect = n - n_correct,
    .groups = "drop"
  )

# ----------------------------------------------------------
# 8. Allometrie – Fehlermaße
# ----------------------------------------------------------

allometrie_stats <- data_eval %>%
  mutate(
    err_h      = chm_mean_m - tr_hgth_m,
    abs_err_h  = abs(err_h),
    err_dbh    = dbh_pred_m - dbh_m,
    abs_err_dbh = abs(err_dbh),
    err_cr     = crown_m2 - cr_area_m2,
    abs_err_cr = abs(err_cr)
  ) %>%
  group_by(method) %>%
  summarise(
    n = n(),
    mae_height_m   = mean(abs_err_h, na.rm = TRUE),
    rmse_height_m  = sqrt(mean(err_h^2, na.rm = TRUE)),
    bias_height_m  = mean(err_h, na.rm = TRUE),
    
    mae_dbh_m      = mean(abs_err_dbh, na.rm = TRUE),
    rmse_dbh_m     = sqrt(mean(err_dbh^2, na.rm = TRUE)),
    bias_dbh_m     = mean(err_dbh, na.rm = TRUE),
    
    mae_crown_m2   = mean(abs_err_cr, na.rm = TRUE),
    rmse_crown_m2  = sqrt(mean(err_cr^2, na.rm = TRUE)),
    bias_crown_m2  = mean(err_cr, na.rm = TRUE),
    .groups = "drop"
  )

allometrie_by_species <- data_eval %>%
  mutate(
    err_h  = chm_mean_m - tr_hgth_m,
    err_dbh = dbh_pred_m - dbh_m,
    err_cr  = crown_m2 - cr_area_m2
  ) %>%
  group_by(method, species_gt) %>%
  summarise(
    n = n(),
    mae_height_m   = mean(abs(err_h), na.rm = TRUE),
    bias_height_m  = mean(err_h, na.rm = TRUE),
    mae_dbh_m      = mean(abs(err_dbh), na.rm = TRUE),
    bias_dbh_m     = mean(err_dbh, na.rm = TRUE),
    mae_crown_m2   = mean(abs(err_cr), na.rm = TRUE),
    bias_crown_m2  = mean(err_cr, na.rm = TRUE),
    .groups = "drop"
  )

# ----------------------------------------------------------
# 9. Korrelationen
# ----------------------------------------------------------

cor_stats <- data_eval %>%
  group_by(method) %>%
  summarise(
    n_pairs_height = sum(!is.na(chm_mean_m) & !is.na(tr_hgth_m)),
    n_pairs_dbh    = sum(!is.na(dbh_pred_m) & !is.na(dbh_m)),
    n_pairs_crown  = sum(!is.na(crown_m2) & !is.na(cr_area_m2)),
    cor_height = if (n_pairs_height > 1) cor(chm_mean_m, tr_hgth_m, use = "complete.obs") else NA_real_,
    cor_dbh    = if (n_pairs_dbh    > 1) cor(dbh_pred_m, dbh_m,       use = "complete.obs") else NA_real_,
    cor_crown  = if (n_pairs_crown  > 1) cor(crown_m2,   cr_area_m2,  use = "complete.obs") else NA_real_,
    .groups = "drop"
  )

# ----------------------------------------------------------
# 10. Plots (optional)
# ----------------------------------------------------------

p_height <- ggplot(
  data_eval %>% mutate(species_gt = factor(species_gt)),
  aes(x = tr_hgth_m, y = chm_mean_m, colour = method)
) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~ species_gt, scales = "free") +
  labs(x = "GT Höhe tr_hgth [m]", y = "CHM mean [m]", colour = "Methode") +
  theme_bw()

p_dbh <- ggplot(
  data_eval %>% mutate(species_gt = factor(species_gt)),
  aes(x = dbh_m, y = dbh_pred_m, colour = method)
) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~ species_gt, scales = "free") +
  labs(x = "GT DBH [m]", y = "Allometrie-DBH [m]", colour = "Methode") +
  theme_bw()

# ----------------------------------------------------------
# 11. Ergebnisse speichern
# ----------------------------------------------------------

write_csv(species_stats,         file.path(out_dir, "pipe2_species_stats.csv"))
write_csv(zustand_stats,         file.path(out_dir, "pipe2_condition_stats.csv"))
write_csv(allometrie_stats,      file.path(out_dir, "pipe2_allometry_overall.csv"))
write_csv(allometrie_by_species, file.path(out_dir, "pipe2_allometry_by_species.csv"))
write_csv(cor_stats,             file.path(out_dir, "pipe2_correlations.csv"))

ggsave(file.path(out_dir, "pipe2_height_scatter_by_species.png"), p_height,
       width = 10, height = 6, dpi = 300)
ggsave(file.path(out_dir, "pipe2_dbh_scatter_by_species.png"), p_dbh,
       width = 10, height = 6, dpi = 300)

cat("PIPE2-Statistiken gespeichert unter:", out_dir, "\n")
