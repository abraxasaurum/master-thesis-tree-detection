#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(sf)
  library(tidyverse)
})

# =========================
# 1) PATHS (EDIT)
# =========================
gt_gpkg <- "/home/abrax/Desktop/Martelo/GT_martel.gpkg"

# GT attribute table that contains GT DBH and GT height by fid
# (you said: CSV has fid but no condition; that's fine)
gt_attr_csv <- "/home/abrax/Desktop/Martelo/martel_MIT_h_perplex.csv"

matches_tcd <- "/home/abrax/Desktop/Martelo/infer_compare_martel/martel_TP_TCD_matches.csv"
matches_dt2 <- "/home/abrax/Desktop/Martelo/infer_compare_martel/martel_TP_Detectree2_matches.csv"

pipe2_tcd <- "/home/abrax/Desktop/Martelo/out_pipe2/martel_tcd_results.csv"
pipe2_dt2 <- "/home/abrax/Desktop/Martelo/out_pipe2/martel_detectree2_results.csv"

output_dir <- "/home/abrax/Desktop/Martelo/infer_compare_martel/statistics_martel"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# 2) GT CODE MAPS
# =========================
map_ba <- function(x) {
  case_when(
    x == 40 ~ "Pine",
    TRUE ~ paste0("BA_", x)
  )
}

# Adapt codes if needed
map_zustand <- function(z) {
  case_when(
    z %in% c(0, 1) ~ "healthy",
    z == 2 ~ "damaged",
    z == 3 ~ "dead",
    TRUE ~ paste0("Z_", z)
  )
}

# =========================
# 3) LOAD GT (species + condition)
# =========================
stopifnot(file.exists(gt_gpkg))
gt_raw <- st_read(gt_gpkg, quiet = TRUE)

# Robust GT fid: use row_number() to mimic QGIS feature id if fid not stored
gt_labels <- gt_raw %>%
  mutate(gt_fid = row_number()) %>%
  st_drop_geometry() %>%
  transmute(
    gt_fid,
    gt_species   = map_ba(as.integer(ba)),
    gt_condition = map_zustand(as.integer(zustand))
  )

# =========================
# 4) LOAD GT ATTRIBUTES (GT DBH + GT HEIGHT)
# =========================
stopifnot(file.exists(gt_attr_csv))
gt_attr <- readr::read_csv(gt_attr_csv, show_col_types = FALSE)

# ---- IMPORTANT ----
# Adjust these column names to your GT-CSV once:
# - fid column
# - GT DBH column (in meters or cm)
# - GT height column (in meters)
#
# Example assumptions (EDIT to match your file):
fid_col       <- "fid"
gt_dbh_col    <- "d 1.3 [cm]"        # could be "tr_dbh" etc.
gt_height_col <- "h [m]"    # you used "GT Höhe tr_hgth [m]" in your plot

gt_attr2 <- gt_attr %>%
  transmute(
    gt_fid = as.integer(.data[[fid_col]]),
    gt_dbh_m = as.numeric(.data[[gt_dbh_col]]),
    gt_height_m = as.numeric(.data[[gt_height_col]])
  )

# If your dbh is in cm, convert here:
 gt_attr2 <- gt_attr2 %>% mutate(gt_dbh_m = gt_dbh_m / 100)

# =========================
# 5) LOAD PIPE2 RESULTS (predictions)
# =========================
stopifnot(file.exists(pipe2_tcd), file.exists(pipe2_dt2))

tcd <- readr::read_csv(pipe2_tcd, show_col_types = FALSE) %>%
  mutate(method = "TCD") %>%
  rename(det_id = id)

dt2 <- readr::read_csv(pipe2_dt2, show_col_types = FALSE) %>%
  mutate(method = "Detectree2") %>%
  rename(det_id = tree_id)

pipe2 <- bind_rows(tcd, dt2) %>%
  mutate(
    method = factor(method, levels = c("Detectree2", "TCD")),
    det_id = as.integer(det_id),
    pred_species = as.character(baumart_pred),
    pred_condition = as.character(zustand_pred),
    tpfp = as.character(tpfp),
    chm_mean = as.numeric(chm_mean),
    # DBH in cm in your outputs; keep both cm and m for convenience
    bhd_height_crown_cm = as.numeric(bhd_height_crown_cm),
    bhd_height_crown_m  = as.numeric(bhd_height_crown_cm) / 100
  )

# =========================
# 6) LOAD MATCHES (TP link det_id -> gt_fid)
# =========================
stopifnot(file.exists(matches_tcd), file.exists(matches_dt2))

m_tcd <- readr::read_csv(matches_tcd, show_col_types = FALSE) %>%
  transmute(method = "TCD",
            det_id = as.integer(det_fid),
            gt_fid = as.integer(gt_fid),
            iou = as.numeric(iou))

m_dt2 <- readr::read_csv(matches_dt2, show_col_types = FALSE) %>%
  transmute(method = "Detectree2",
            det_id = as.integer(det_fid),
            gt_fid = as.integer(gt_fid),
            iou = as.numeric(iou))

matches <- bind_rows(m_tcd, m_dt2) %>%
  mutate(method = factor(method, levels = c("Detectree2", "TCD")))

# =========================
# 7) BUILD TP TABLE WITH GT LABELS + GT ATTRS + PRED
# =========================
tp_tbl <- pipe2 %>%
  filter(tpfp == "TP") %>%
  inner_join(matches, by = c("method", "det_id")) %>%
  left_join(gt_labels, by = "gt_fid") %>%
  left_join(gt_attr2, by = "gt_fid") %>%
  filter(!is.na(gt_fid))

# =========================
# 8) PLOT 1: GT vs Pred – Species + Condition (main-text style)
#    (TP subset only, because only TP can be linked to GT condition)
# =========================
# Keep it simple: 2 panels (GT vs Pred), bars by condition, facets by species, color by method
cond_order <- c("healthy", "damaged", "dead")

p_cond <- tp_tbl %>%
  mutate(
    gt_condition = factor(gt_condition, levels = cond_order),
    pred_condition = factor(pred_condition, levels = cond_order),
    gt_species = factor(gt_species, levels = c("Pine", "Douglas Fir", "Beech")),
    pred_species = factor(pred_species, levels = c("Pine", "Douglas Fir", "Beech"))
  ) %>%
  select(method, gt_species, gt_condition, pred_species, pred_condition) %>%
  pivot_longer(
    cols = c(gt_condition, pred_condition),
    names_to = "source",
    values_to = "condition"
  ) %>%
  mutate(source = recode(source, gt_condition = "GT", pred_condition = "Pred")) %>%
  # use GT species for facet to stay consistent (TP linked to GT)
  ggplot(aes(x = condition, fill = method)) +
  geom_bar(position = position_dodge(width = 0.8), width = 0.7) +
  facet_grid(source ~ gt_species) +
  labs(
    title = "Marteloscope (TP subset) – Condition distribution: GT vs predicted",
    subtitle = "Only true positives are shown (detections matched to GT via IoU).",
    x = "Condition",
    y = "Count",
    fill = "Method"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom",
    strip.text = element_text(face = "bold")
  )

ggsave(file.path(output_dir, "09_gt_vs_pred_condition_bars.png"),
       p_cond, width = 12, height = 6, dpi = 300)

# =========================
# 9) PLOT 2: Scatter – GT DBH vs Allometric DBH (like main text)
# =========================
p_dbh <- tp_tbl %>%
  filter(!is.na(gt_dbh_m), !is.na(bhd_height_crown_m),
         gt_dbh_m > 0, bhd_height_crown_m > 0) %>%
  mutate(gt_species = factor(gt_species, levels = c("Beech", "Douglas Fir", "Pine"))) %>%
  ggplot(aes(x = gt_dbh_m, y = bhd_height_crown_m, color = method)) +
  geom_point(alpha = 0.5, size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~ gt_species, scales = "free") +
  labs(
    title = "Allometric DBH vs GT DBH (Marteloscope, TP subset)",
    x = "GT DBH [m]",
    y = "Allometric DBH [m]",
    color = "Method"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right",
    strip.text = element_text(face = "bold")
  )

ggsave(file.path(output_dir, "10_pipe2_dbh_scatter_by_species.png"),
       p_dbh, width = 12, height = 7, dpi = 300)

# =========================
# 10) PLOT 3: Scatter – GT Height vs CHM mean (like main text)
# =========================
p_h <- tp_tbl %>%
  filter(!is.na(gt_height_m), !is.na(chm_mean),
         gt_height_m > 0, chm_mean > 0) %>%
  mutate(gt_species = factor(gt_species, levels = c("Beech", "Douglas Fir", "Pine"))) %>%
  ggplot(aes(x = gt_height_m, y = chm_mean, color = method)) +
  geom_point(alpha = 0.5, size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~ gt_species, scales = "free") +
  labs(
    title = "CHM height vs GT height (Marteloscope, TP subset)",
    x = "GT height [m]",
    y = "CHM mean [m]",
    color = "Method"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right",
    strip.text = element_text(face = "bold")
  )

ggsave(file.path(output_dir, "11_pipe2_height_scatter_by_species.png"),
       p_h, width = 12, height = 7, dpi = 300)

cat("✅ Done.\n")
cat("Saved:\n")
cat(" - 09_gt_vs_pred_condition_bars.png\n")
cat(" - 10_pipe2_dbh_scatter_by_species.png\n")
cat(" - 11_pipe2_height_scatter_by_species.png\n")
