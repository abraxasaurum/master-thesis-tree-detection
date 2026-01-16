#!/usr/bin/env Rscript

# ═════════════════════════════════════════════════════════════════════
# 📊 MARTEL – INFERENCE COMPARISON (ROBUST VERSION)
# Works with:
#   1) method-level summary (e.g., martel_summary_all_methods.csv)
#   2) optionally richer per-tile / per-species / gt_set tables (if present)
#
# Key idea:
# - If your CSV contains ONLY method-level rows (as in your output),
#   the script will still generate valid tables + plots (no tile/species facets).
# - If columns like tile/species/gt_set exist, it will automatically stratify.
#
# NOTE on partial/incomplete GT:
# - Precision/FP/F1 can be "apparent" (biased low) when GT is incomplete.
#   The script labels this clearly in outputs when eval_mode="partial_gt".
# ═════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(scales)
})

# ═══ 1) CONFIG ═══
input_csv  <- "/home/abrax/Desktop/Martelo/infer_compare_martel/martel_summary_all_methods.csv"  # adjust if needed
output_dir <- "/home/abrax/Desktop/Infer_stat_input/inference_comparison/statistics_martel"       # adjust if needed
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Evaluation mode:
# - "complete_gt"  : classic interpretation
# - "partial_gt"   : GT has gaps; precision/FP/F1 are *apparent* (biased)
eval_mode <- "partial_gt"

# Plot palette
method_colors <- c("TCD" = "#E69F00", "Detectree2" = "#56B4E9", "DeepTree" = "#009E73")

# ═══ 2) LOAD DATA ═══
df <- readr::read_csv(input_csv, show_col_types = FALSE)

cat("✓ Loaded data:\n")
print(df)
cat(sprintf("\n✓ Rows: %d | Cols: %d\n", nrow(df), ncol(df)))

if (nrow(df) == 0) stop("❌ CSV is empty. Check file path/content.")

# ═══ 3) NORMALIZE COLUMN NAMES (make script tolerant) ═══
# Expect at minimum:
# method, TP, FP, FN, Recall, Precision, F1, mean_IoU (or mean_IoU-like)
required_min <- c("method")

missing_min <- setdiff(required_min, names(df))
if (length(missing_min) > 0) stop(paste("❌ Missing required columns:", paste(missing_min, collapse=", ")))

# Handle common naming variants
if (!"mean_IoU" %in% names(df)) {
  # try alternatives
  alt <- intersect(names(df), c("mean_iou", "Mean_IoU", "iou_mean", "IoU", "meanIoU"))
  if (length(alt) > 0) df <- df %>% rename(mean_IoU = all_of(alt[1]))
}

# Optional grouping columns (exist only in richer tables)
has_gt_set <- "gt_set" %in% names(df)
has_tile   <- "tile"   %in% names(df)
has_species<- "species"%in% names(df)

# Optional totals (present in your martel summary)
has_gt_total  <- "gt_total"  %in% names(df)
has_det_total <- "det_total" %in% names(df)

# Add gt_set default only if it truly exists in your design; otherwise keep absent.
# (Do NOT force 'train' when it isn't present; that caused confusion before.)
if (has_gt_set) {
  df <- df %>% mutate(gt_set = as.character(gt_set))
}

# Label metrics in partial GT mode
metric_note <- if (eval_mode == "partial_gt") {
  "Note: GT is incomplete. Precision/FP/F1 are *apparent* and biased low; Recall is relative to annotated GT."
} else {
  "Note: Classic evaluation (assumes GT complete within AOI)."
}

cat("\n", metric_note, "\n\n")

# ═══ 4) DESCRIPTIVE TABLES ═══

# Helper: safe mean/sd for tiny n
safe_sd <- function(x) if (length(na.omit(x)) >= 2) sd(x, na.rm = TRUE) else NA_real_

# 4.1 By method (works for both summary-only and richer data)
stats_method <- df %>%
  group_by(method) %>%
  summarise(
    n_rows = n(),
    # If summary-only, these are already single values; mean == value.
    mean_Recall     = round(mean(Recall, na.rm = TRUE), 3),
    sd_Recall       = round(safe_sd(Recall), 3),
    mean_Precision  = round(mean(Precision, na.rm = TRUE), 3),
    sd_Precision    = round(safe_sd(Precision), 3),
    mean_F1         = round(mean(F1, na.rm = TRUE), 3),
    sd_F1           = round(safe_sd(F1), 3),
    mean_IoU        = round(mean(mean_IoU, na.rm = TRUE), 3),
    sd_IoU          = round(safe_sd(mean_IoU), 3),
    total_TP        = sum(TP, na.rm = TRUE),
    total_FP        = sum(FP, na.rm = TRUE),
    total_FN        = sum(FN, na.rm = TRUE),
    gt_total        = if (has_gt_total)  max(gt_total,  na.rm = TRUE) else NA_real_,
    det_total       = if (has_det_total) max(det_total, na.rm = TRUE) else NA_real_,
    .groups = "drop"
  )

cat("👉 Statistics by method:\n")
print(stats_method)
readr::write_csv(stats_method, file.path(output_dir, "statistics_by_method.csv"))

# 4.2 By method + gt_set (only if gt_set exists)
if (has_gt_set) {
  stats_method_set <- df %>%
    group_by(method, gt_set) %>%
    summarise(
      n_rows = n(),
      mean_Recall     = round(mean(Recall, na.rm = TRUE), 3),
      sd_Recall       = round(safe_sd(Recall), 3),
      mean_Precision  = round(mean(Precision, na.rm = TRUE), 3),
      sd_Precision    = round(safe_sd(Precision), 3),
      mean_F1         = round(mean(F1, na.rm = TRUE), 3),
      sd_F1           = round(safe_sd(F1), 3),
      mean_IoU        = round(mean(mean_IoU, na.rm = TRUE), 3),
      sd_IoU          = round(safe_sd(mean_IoU), 3),
      total_TP        = sum(TP, na.rm = TRUE),
      total_FP        = sum(FP, na.rm = TRUE),
      total_FN        = sum(FN, na.rm = TRUE),
      .groups = "drop"
    )
  
  cat("\n👉 Statistics by method and gt_set:\n")
  print(stats_method_set)
  readr::write_csv(stats_method_set, file.path(output_dir, "statistics_by_method_and_set.csv"))
}

# 4.3 By tile/species (only if these columns exist)
if (has_tile && has_species) {
  stats_tile <- df %>%
    group_by(tile, species) %>%
    summarise(
      n_rows = n(),
      mean_Recall    = round(mean(Recall, na.rm = TRUE), 3),
      mean_Precision = round(mean(Precision, na.rm = TRUE), 3),
      mean_F1        = round(mean(F1, na.rm = TRUE), 3),
      mean_IoU       = round(mean(mean_IoU, na.rm = TRUE), 3),
      .groups = "drop"
    )
  
  cat("\n👉 Statistics by tile/species:\n")
  print(stats_tile)
  readr::write_csv(stats_tile, file.path(output_dir, "statistics_by_tile.csv"))
  
  if (has_gt_set) {
    stats_tile_set <- df %>%
      group_by(tile, species, gt_set) %>%
      summarise(
        n_rows = n(),
        mean_Recall    = round(mean(Recall, na.rm = TRUE), 3),
        mean_Precision = round(mean(Precision, na.rm = TRUE), 3),
        mean_F1        = round(mean(F1, na.rm = TRUE), 3),
        mean_IoU       = round(mean(mean_IoU, na.rm = TRUE), 3),
        .groups = "drop"
      )
    readr::write_csv(stats_tile_set, file.path(output_dir, "statistics_by_tile_and_set.csv"))
  }
}

# ═══ 5) VISUALIZATIONS ═══

# Prepare long form for metric bars (method-level)
df_plot <- stats_method %>%
  select(method, mean_Recall, mean_Precision, mean_F1, mean_IoU, total_TP, total_FP, total_FN) %>%
  mutate(method = factor(method, levels = names(method_colors)))

# 5.1 Bar chart: Recall/Precision/F1/IoU by method
df_metrics_long <- df_plot %>%
  select(method, mean_Recall, mean_Precision, mean_F1, mean_IoU) %>%
  pivot_longer(cols = -method, names_to = "Metric", values_to = "Value") %>%
  mutate(
    Metric = recode(Metric,
                    "mean_Recall"    = "Recall",
                    "mean_Precision" = if (eval_mode == "partial_gt") "Precision (apparent)" else "Precision",
                    "mean_F1"        = if (eval_mode == "partial_gt") "F1 (apparent)" else "F1",
                    "mean_IoU"       = "Mean IoU (TP)"
    )
  )

p1 <- ggplot(df_metrics_long, aes(x = method, y = Value, fill = method)) +
  geom_col(width = 0.7) +
  facet_wrap(~ Metric, ncol = 2, scales = "free_y") +
  scale_fill_manual(values = method_colors, drop = FALSE) +
  scale_y_continuous(limits = c(0, 1), oob = scales::squish) +
  labs(
    title = "Martel – Detection Metrics by Method",
    subtitle = metric_note,
    x = "Method",
    y = "Score"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none", plot.title = element_text(face = "bold"))

ggsave(file.path(output_dir, "01_metrics_by_method.png"), p1, width = 10, height = 7, dpi = 300)
cat("✓ Saved: 01_metrics_by_method.png\n")

# 5.2 TP/FP/FN stacked bars by method
df_counts_long <- df_plot %>%
  select(method, total_TP, total_FP, total_FN) %>%
  pivot_longer(cols = -method, names_to = "Category", values_to = "Count") %>%
  mutate(
    Category = recode(Category,
                      "total_TP" = "TP",
                      "total_FP" = if (eval_mode == "partial_gt") "FP (includes unlabeled true trees)" else "FP",
                      "total_FN" = "FN"
    )
  )

p2 <- ggplot(df_counts_long, aes(x = method, y = Count, fill = Category)) +
  geom_col(width = 0.7) +
  labs(
    title = "Martel – Outcome Counts by Method",
    subtitle = metric_note,
    x = "Method",
    y = "Count",
    fill = "Category"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(output_dir, "02_tp_fp_fn_by_method.png"), p2, width = 10, height = 6, dpi = 300)
cat("✓ Saved: 02_tp_fp_fn_by_method.png\n")

# 5.3 If gt_total and det_total exist: detection density overview
if (has_gt_total && has_det_total) {
  p3 <- stats_method %>%
    mutate(method = factor(method, levels = names(method_colors))) %>%
    select(method, gt_total, det_total, total_TP) %>%
    pivot_longer(cols = c(gt_total, det_total, total_TP), names_to = "Type", values_to = "Count") %>%
    mutate(Type = recode(Type, gt_total="GT total (annotated)", det_total="Detections total", total_TP="TP")) %>%
    ggplot(aes(x = method, y = Count, fill = method)) +
    geom_col(width = 0.7) +
    facet_wrap(~Type, scales = "free_y") +
    scale_fill_manual(values = method_colors, drop = FALSE) +
    labs(
      title = "Martel – Totals Overview",
      subtitle = "GT totals are annotated subset (not full stand) when GT is incomplete.",
      x = "Method",
      y = "Count"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none", plot.title = element_text(face = "bold"))
  
  ggsave(file.path(output_dir, "03_totals_overview.png"), p3, width = 10, height = 7, dpi = 300)
  cat("✓ Saved: 03_totals_overview.png\n")
}





# ═══════════════════════════════════════════════════════════════
# 5.4 | IoU DISTRIBUTION ANALYSIS (from martel_TP_ALL_COMBINED.csv)
# - Uses the TP match table to analyze IoU distributions per method
# - Produces: violin/boxplot, ECDF, threshold sensitivity, summary stats
# ═══════════════════════════════════════════════════════════════

tp_csv <- file.path(dirname(input_csv), "martel_TP_ALL_COMBINED.csv")
# Alternative (falls du es woanders gespeichert hast):
# tp_csv <- "/home/abrax/Desktop/Martelo/infer_compare_martel/martel_TP_ALL_COMBINED.csv"

if (file.exists(tp_csv)) {
  
  tp <- readr::read_csv(tp_csv, show_col_types = FALSE)
  cat(sprintf("\n✓ Loaded TP match table: %s (%d rows)\n", basename(tp_csv), nrow(tp)))
  
  # Minimal validation
  req_cols <- c("method", "iou")
  miss <- setdiff(req_cols, names(tp))
  if (length(miss) > 0) stop(paste("❌ TP table missing columns:", paste(miss, collapse=", ")))
  
  tp <- tp %>%
    mutate(
      method = as.factor(method),
      iou = as.numeric(iou)
    ) %>%
    filter(!is.na(iou), iou >= 0, iou <= 1)
  
  # ---------- 5.4.1 Summary stats per method ----------
  iou_stats <- tp %>%
    group_by(method) %>%
    summarise(
      n_matches = n(),
      mean_iou  = round(mean(iou), 3),
      sd_iou    = round(sd(iou), 3),
      q05       = round(quantile(iou, 0.05), 3),
      q25       = round(quantile(iou, 0.25), 3),
      median    = round(quantile(iou, 0.50), 3),
      q75       = round(quantile(iou, 0.75), 3),
      q95       = round(quantile(iou, 0.95), 3),
      min_iou   = round(min(iou), 3),
      max_iou   = round(max(iou), 3),
      .groups = "drop"
    )
  
  cat("\n👉 IoU summary statistics by method:\n")
  print(iou_stats)
  readr::write_csv(iou_stats, file.path(output_dir, "statistics_iou_by_method.csv"))
  
  # ---------- 5.4.2 Violin + boxplot of IoU per method ----------
  p_iou_violin <- ggplot(tp, aes(x = method, y = iou, fill = method)) +
    geom_violin(trim = TRUE, alpha = 0.65) +
    geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.85) +
    geom_jitter(width = 0.08, alpha = 0.25, size = 1) +
    scale_fill_manual(values = method_colors, drop = FALSE) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(
      title = "Martel – IoU distribution of matched TP pairs",
      subtitle = "IoU values from polygon-to-polygon matching (TP table)",
      x = "Method",
      y = "IoU (TP matches)"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold")
    )
  
  ggsave(file.path(output_dir, "04_iou_violin_by_method.png"),
         p_iou_violin, width = 10, height = 6, dpi = 300)
  cat("✓ Saved: 04_iou_violin_by_method.png\n")
  
  # ---------- 5.4.3 ECDF curves (cumulative distribution) ----------
  p_iou_ecdf <- ggplot(tp, aes(x = iou, color = method)) +
    stat_ecdf(size = 1.1) +
    scale_color_manual(values = method_colors, drop = FALSE) +
    scale_x_continuous(limits = c(0, 1)) +
    labs(
      title = "Martel – ECDF of IoU for TP matches",
      subtitle = "Lower curve indicates higher IoUs (more mass at high overlap)",
      x = "IoU",
      y = "ECDF (fraction ≤ IoU)",
      color = "Method"
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))
  
  ggsave(file.path(output_dir, "05_iou_ecdf_by_method.png"),
         p_iou_ecdf, width = 10, height = 6, dpi = 300)
  cat("✓ Saved: 05_iou_ecdf_by_method.png\n")
  
  # ---------- 5.4.4 Threshold sensitivity: TP count vs IoU threshold ----------
  thresholds <- seq(0.10, 0.90, by = 0.05)
  thr_tbl <- expand_grid(method = levels(tp$method), thr = thresholds) %>%
    left_join(tp, by = "method") %>%
    group_by(method, thr) %>%
    summarise(tp_ge_thr = sum(iou >= thr, na.rm = TRUE), .groups = "drop")
  
  p_thr <- ggplot(thr_tbl, aes(x = thr, y = tp_ge_thr, color = method)) +
    geom_line(size = 1.1) +
    geom_point(size = 2) +
    scale_color_manual(values = method_colors, drop = FALSE) +
    labs(
      title = "Martel – TP matches retained vs IoU threshold",
      subtitle = "How strict overlap criteria affect retained TP matches",
      x = "IoU threshold",
      y = "TP matches with IoU ≥ threshold",
      color = "Method"
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))
  
  ggsave(file.path(output_dir, "06_tp_retained_vs_iou_threshold.png"),
         p_thr, width = 10, height = 6, dpi = 300)
  cat("✓ Saved: 06_tp_retained_vs_iou_threshold.png\n")
  
} else {
  cat(sprintf("\n⚠️ TP table not found: %s\n", tp_csv))
  cat("   Skipping IoU distribution analysis.\n")
}






# ═══════════════════════════════════════════════════════════════
# 5.5 | IoU COMPARISON: Significance + Effect Size + Threshold rates + Bootstrap CI
# Requires: object `tp` already loaded from martel_TP_ALL_COMBINED.csv
# Outputs:
#   - statistics_iou_tests.csv
#   - statistics_iou_threshold_rates.csv
#   - statistics_iou_bootstrap_ci.csv
#   - 07_iou_threshold_rates.png
# ═══════════════════════════════════════════════════════════════

if (exists("tp")) {
  
  # --- 5.5.1 Pairwise Wilcoxon (Mann–Whitney) test for IoU ---
  methods_present <- sort(unique(as.character(tp$method)))
  if (length(methods_present) >= 2) {
    
    # helper: Cliff's delta (no packages)
    cliffs_delta <- function(x, y) {
      x <- x[!is.na(x)]; y <- y[!is.na(y)]
      if (length(x) == 0 || length(y) == 0) return(NA_real_)
      # Efficient rank-based implementation:
      # delta = (2*U)/(n_x*n_y) - 1, where U is Mann-Whitney U for x
      r <- rank(c(x, y))
      nx <- length(x); ny <- length(y)
      rx <- sum(r[1:nx])
      Ux <- rx - nx*(nx+1)/2
      delta <- (2*Ux)/(nx*ny) - 1
      as.numeric(delta)
    }
    
    # Build pairwise comparison table (for Martel typically only TCD vs Detectree2)
    combs <- t(combn(methods_present, 2))
    test_tbl <- purrr::map_dfr(seq_len(nrow(combs)), function(i) {
      a <- combs[i, 1]; b <- combs[i, 2]
      xa <- tp %>% filter(method == a) %>% pull(iou)
      xb <- tp %>% filter(method == b) %>% pull(iou)
      
      # Wilcoxon rank-sum test (unpaired)
      wt <- wilcox.test(xa, xb, alternative = "two.sided", exact = FALSE)
      
      tibble(
        method_A = a,
        method_B = b,
        n_A = sum(!is.na(xa)),
        n_B = sum(!is.na(xb)),
        median_A = round(median(xa, na.rm = TRUE), 3),
        median_B = round(median(xb, na.rm = TRUE), 3),
        mean_A   = round(mean(xa, na.rm = TRUE), 3),
        mean_B   = round(mean(xb, na.rm = TRUE), 3),
        p_value  = wt$p.value,
        p_adj_bh = p.adjust(wt$p.value, method = "BH"),
        cliffs_delta = round(cliffs_delta(xa, xb), 3)
      )
    })
    
    readr::write_csv(test_tbl, file.path(output_dir, "statistics_iou_tests.csv"))
    cat("✓ Saved: statistics_iou_tests.csv\n")
    
    # --- 5.5.2 Threshold rates: fraction of TP matches with IoU >= threshold ---
    thr_vec <- c(0.30, 0.50, 0.70, 0.75, 0.80, 0.85)
    
    thr_rates <- tp %>%
      group_by(method) %>%
      summarise(
        n_matches = n(),
        rate_iou_ge_0_30 = mean(iou >= 0.30, na.rm = TRUE),
        rate_iou_ge_0_50 = mean(iou >= 0.50, na.rm = TRUE),
        rate_iou_ge_0_70 = mean(iou >= 0.70, na.rm = TRUE),
        rate_iou_ge_0_75 = mean(iou >= 0.75, na.rm = TRUE),
        rate_iou_ge_0_80 = mean(iou >= 0.80, na.rm = TRUE),
        rate_iou_ge_0_85 = mean(iou >= 0.85, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(across(starts_with("rate_"), ~round(.x, 3)))
    
    readr::write_csv(thr_rates, file.path(output_dir, "statistics_iou_threshold_rates.csv"))
    cat("✓ Saved: statistics_iou_threshold_rates.csv\n")
    
    # Plot (tidy form)
    thr_rates_long <- tp %>%
      mutate(method = as.character(method)) %>%
      tidyr::expand_grid(thr = thr_vec) %>%
      group_by(method, thr) %>%
      summarise(rate = mean(iou >= thr, na.rm = TRUE), .groups = "drop")
    
    p_thr_rates <- ggplot(thr_rates_long, aes(x = thr, y = rate, color = method)) +
      geom_line(size = 1.1) +
      geom_point(size = 2) +
      scale_color_manual(values = method_colors, drop = FALSE) +
      scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
      labs(
        title = "Martel – Share of TP matches above IoU thresholds",
        subtitle = "Higher curve indicates more geometrically accurate TP matches",
        x = "IoU threshold",
        y = "Share of TP matches (IoU ≥ threshold)",
        color = "Method"
      ) +
      theme_minimal(base_size = 12) +
      theme(plot.title = element_text(face = "bold"))
    
    ggsave(file.path(output_dir, "07_iou_threshold_rates.png"),
           p_thr_rates, width = 10, height = 6, dpi = 300)
    cat("✓ Saved: 07_iou_threshold_rates.png\n")
    
    
    # --- 5.5.3 Bootstrap CI for mean and median IoU per method ---
    set.seed(1)
    B <- 2000
    
    boot_ci <- function(x, fun = median, B = 2000, conf = 0.95) {
      x <- x[!is.na(x)]
      if (length(x) < 5) return(c(NA, NA, NA))
      stats <- replicate(B, fun(sample(x, replace = TRUE)))
      alpha <- (1 - conf)/2
      c(
        estimate = fun(x),
        lo = unname(quantile(stats, alpha)),
        hi = unname(quantile(stats, 1 - alpha))
      )
    }
    
    boot_tbl <- tp %>%
      group_by(method) %>%
      summarise(
        n = n(),
        mean_est   = mean(iou, na.rm = TRUE),
        mean_lo    = boot_ci(iou, fun = mean,   B = B)[2],
        mean_hi    = boot_ci(iou, fun = mean,   B = B)[3],
        median_est = median(iou, na.rm = TRUE),
        median_lo  = boot_ci(iou, fun = median, B = B)[2],
        median_hi  = boot_ci(iou, fun = median, B = B)[3],
        .groups = "drop"
      ) %>%
      mutate(across(where(is.numeric), ~round(.x, 3)))
    
    readr::write_csv(boot_tbl, file.path(output_dir, "statistics_iou_bootstrap_ci.csv"))
    cat("✓ Saved: statistics_iou_bootstrap_ci.csv\n")
    
  } else {
    cat("⚠️ Not enough methods in TP table for pairwise IoU comparison.\n")
  }
  
} else {
  cat("⚠️ Object `tp` not found. Run IoU block (5.4) first.\n")
}











# ═══ 6) STATISTICAL TESTS (only if enough replicates) ═══
# With method-level summary (1 row per method), tests are not meaningful.
# If you later feed a per-tile/per-species table with multiple rows per method, tests will run.

run_wilcox <- function(d, metric, a="TCD", b="Detectree2") {
  if (!all(c("method", metric) %in% names(d))) return(NULL)
  x <- d %>% filter(method == a) %>% pull(all_of(metric))
  y <- d %>% filter(method == b) %>% pull(all_of(metric))
  x <- x[!is.na(x)]; y <- y[!is.na(y)]
  if (length(x) < 2 || length(y) < 2) return(NULL)
  wilcox.test(x, y)
}

# If there are multiple rows per method (e.g., tiles), try tests
if (df %>% count(method) %>% pull(n) %>% max() >= 2) {
  cat("\n🔬 Wilcoxon tests (if replicates exist):\n")
  for (m in c("Recall", "Precision", "F1", "mean_IoU")) {
    res <- run_wilcox(df, m)
    if (!is.null(res)) {
      cat(sprintf(" - %s: p=%.6f\n", m, res$p.value))
    } else {
      cat(sprintf(" - %s: skipped (not enough replicates)\n", m))
    }
  }
} else {
  cat("\n🔬 Statistical tests skipped: only 1 row per method (summary-only input).\n")
}

# ═══ 7) DONE ═══
cat("\n✅ Analysis finished.\n")
cat(sprintf("📂 Output directory: %s\n", output_dir))
cat("Files:\n")
cat(" - statistics_by_method.csv\n")
if (has_gt_set) cat(" - statistics_by_method_and_set.csv\n")
if (has_tile && has_species) cat(" - statistics_by_tile.csv (+ optional _and_set)\n")
cat(" - 01_metrics_by_method.png\n")
cat(" - 02_tp_fp_fn_by_method.png\n")
if (has_gt_total && has_det_total) cat(" - 03_totals_overview.png\n")
