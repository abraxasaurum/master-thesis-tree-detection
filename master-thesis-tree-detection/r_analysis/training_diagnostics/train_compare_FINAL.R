#!/usr/bin/env Rscript

# ═══════════════════════════════════════════════════════════════════════════
# 🎓 MASTERARBEIT - TRAINING ANALYSIS MIT DEEPTREES FIX
# ═══════════════════════════════════════════════════════════════════════════
#
# ✅ DeepTrees FIX: Epochen aus CSV-Struktur extrahieren
#    - Epochenzeilen als Anker nutzen
#    - train/val im lokalen 5er-Fenster suchen
#    - Pragmatisch & robust
#
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
# 📦 PACKAGES
# ─────────────────────────────────────────────────────────────────────────
required_packages <- c(
  "tidyverse", "readr", "caret", "patchwork", "scales",
  "ggpubr", "zoo", "gridExtra"
)

install_if_missing <- function(pkgs) {
  to_install <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
  if(length(to_install)) install.packages(to_install, dependencies = TRUE)
}

install_if_missing(required_packages)

suppressPackageStartupMessages(
  invisible(lapply(required_packages, function(pkg) {
    library(pkg, character.only = TRUE)
  }))
)

cat("╔═══════════════════════════════════════════════════════════════╗\n")
cat("║  🎓 MASTERARBEIT - TRAINING ANALYSIS (WITH DEEPTREES)        ║\n")
cat("║     TCD vs Detectree2 vs DeepTrees                           ║\n")
cat("╚═══════════════════════════════════════════════════════════════╝\n\n")

# ─────────────────────────────────────────────────────────────────────────
# 1. DATEN LADEN
# ─────────────────────────────────────────────────────────────────────────

tcd_metrics_path <- "/home/abrax/Desktop/TCD_output/logs/Comparison_9params_TCD_final/Unet-resnet34_9params_e90_lr0.0001_w256_bs4_seed42/metrics.csv"
detectree2_metrics_path <- "/home/abrax/detectree2/training/outputs_single_class_with_final/metrics_detectree2.csv"
deeptrees_metrics_path <- "/home/abrax/Desktop/dp_output/metrics.csv"

output_dir <- "/home/abrax/Desktop/Masterarbeit_Analysis"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("📂 Lade Training Metrics...\n")

# ─────────────────────────────────────────────────────────────────────────
# TCD METRICS
# ─────────────────────────────────────────────────────────────────────────

tcd_raw <- read_csv(tcd_metrics_path, show_col_types = FALSE)

tcd_train <- tcd_raw %>%
  filter(!is.na(train_loss)) %>%
  select(epoch, train_loss, train_iou_mask, lr = `lr-Adam`) %>%
  rename(train_iou = train_iou_mask)

tcd_val <- tcd_raw %>%
  filter(!is.na(val_loss)) %>%
  select(epoch, val_loss, val_iou_mask) %>%
  rename(val_iou = val_iou_mask)

tcd <- full_join(tcd_train, tcd_val, by = "epoch") %>%
  mutate(method = "TCD") %>%
  filter(!is.na(epoch)) %>%
  arrange(epoch)

cat("✓ TCD geladen:", nrow(tcd), "Epochen\n")

# ─────────────────────────────────────────────────────────────────────────
# DETECTREE2 – EPOCHEN AUS ITERATIONEN (90 PSEUDO-EPOCHEN)
# ─────────────────────────────────────────────────────────────────────────
n_epochs <- 90
dt2_raw <- read_csv(detectree2_metrics_path, show_col_types = FALSE)

# ✅ BERECHNE EPOCHE BASIEREND AUF ITERATION
# Wir wissen: MAX_ITER = 3000 (90 Epochen × ~33 Iter/Epoch)
max_iter <- max(dt2_raw$iteration, na.rm = TRUE)
iters_per_epoch <- max_iter / n_epochs

cat("ℹ️  Detectree2: MAX_ITER =", max_iter, ", Iters/Epoch =", round(iters_per_epoch, 1), "\n")

detectree2 <- dt2_raw %>%
  mutate(epoch = ceiling(iteration / iters_per_epoch)) %>%
  group_by(epoch) %>%
  summarise(
    train_loss = mean(total_loss, na.rm = TRUE),
    val_loss = mean(total_val_loss, na.rm = TRUE),
    lr = mean(lr, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(method = "Detectree2") %>%
  filter(epoch <= n_epochs)  # Sicherstelle max 90 Epochen

cat("✓ Detectree2 geladen:", nrow(detectree2), "Epochen\n")


######
#n_epochs <- 90
#dt2_raw <- read_csv(detectree2_metrics_path, show_col_types = FALSE)

#cat("\n📋 Detectree2 RAW Zeilen:", nrow(dt2_raw), "\n")

#max_iter <- max(dt2_raw$iteration, na.rm = TRUE)
#iters_per_epoch <- max_iter / n_epochs
#cat("  Maximale Iteration:", max_iter, "\n")
#cat("  Iterationen pro Epoche (approx.):", iters_per_epoch, "\n\n")

#detectree2 <- dt2_raw %>%
  # Nur Trainings-Iterationen (total_loss vorhanden)
#  filter(!is.na(total_loss)) %>%
#  mutate(
#    epoch = pmin(
#      n_epochs,
#      ceiling(iteration / iters_per_epoch)  # z.B. 1–33 -> Epoche 1, 34–66 -> 2, ...
#    )
#  ) %>%
#  group_by(epoch) %>%
#  summarise(
#    train_loss = mean(total_loss, na.rm = TRUE),
#    val_loss   = if (all(is.na(total_val_loss))) NA_real_ else mean(total_val_loss, na.rm = TRUE),
#    train_iou  = if (all(is.na(train_iou)))      NA_real_ else mean(train_iou, na.rm = TRUE),
#    val_iou    = if (all(is.na(val_iou)))        NA_real_ else mean(val_iou, na.rm = TRUE),
#    lr         = mean(lr, na.rm = TRUE),
#    .groups = "drop"
#  ) %>%
#  mutate(method = "Detectree2") %>%
#  arrange(epoch)

cat("✓ Detectree2 geladen:", nrow(detectree2), "Epochen\n")
cat("  - Train-Loss NAs:", sum(is.na(detectree2$train_loss)), "\n")
cat("  - Val-Loss NAs:",   sum(is.na(detectree2$val_loss)), "\n")
cat("  - Train-IoU NAs:",  sum(is.na(detectree2$train_iou)), "\n")
cat("  - Val-IoU NAs:",    sum(is.na(detectree2$val_iou)), "\n")
######

# ─────────────────────────────────────────────────────────────────────────
# DEEPTREES - PRAGMATISCHER FIX (lokales Fenster)
# ─────────────────────────────────────────────────────────────────────────

deeptrees_raw <- read_csv(deeptrees_metrics_path, show_col_types = FALSE)

cat("\n📋 DeepTrees RAW:\n")
cat("  Total Zeilen:", nrow(deeptrees_raw), "\n")
cat("  Distinct epoch:", n_distinct(deeptrees_raw$epoch), "\n")
cat("  Distinct step:",  n_distinct(deeptrees_raw$step),  "\n\n")

deeptrees <- deeptrees_raw %>%
  transmute(
    epoch      = step,                 # x‑Achse = 0..89
    train_loss = `train/loss`,
    val_loss   = `val/loss`,
    train_iou  = `train/iou`,
    val_iou    = `val/iou`,
    lr         = `lr-Adam`,
    method     = "DeepTrees"
  ) %>%
  arrange(epoch)

cat("✓ DeepTrees geladen:", nrow(deeptrees), "Zeilen (sollten 90 sein)\n")


###############
#cat("\n📋 DeepTrees RAW:\n")
#cat("  Total Zeilen:", nrow(deeptrees_raw), "\n")
#cat("  Zeilen mit epoch:", sum(!is.na(deeptrees_raw$epoch)), "\n\n")

# Epochen-Zeilen als Anker
#epoch_rows <- which(!is.na(deeptrees_raw$epoch))

# Funktion: Für jede Epochenzeile, hole train/val aus lokalem 5er-Fenster
#extract_epoch_block <- function(idx) {
  # Indizes: idx ± 2
#  rows <- (idx-2):(idx+2)
#  rows <- rows[rows > 0 & rows <= nrow(deeptrees_raw)]
  
#  block <- deeptrees_raw[rows, ]
  
  # train = letzte Zeile mit train/loss nicht NA
#  train_row <- block %>%
 #   filter(!is.na(`train/loss`)) %>%
  #  slice_tail(n = 1)
  
  # val = letzte Zeile mit val/loss nicht NA
  #val_row <- block %>%
   # filter(!is.na(`val/loss`)) %>%
    #slice_tail(n = 1)
  
  #tibble(
   # epoch = deeptrees_raw$epoch[idx],
  #  train_loss = if(nrow(train_row) > 0) train_row$`train/loss`[1] else NA_real_,
  #  val_loss = if(nrow(val_row) > 0) val_row$`val/loss`[1] else NA_real_,
  #  train_iou = if(nrow(train_row) > 0) train_row$`train/iou`[1] else NA_real_,
  #  val_iou = if(nrow(val_row) > 0) val_row$`val/iou`[1] else NA_real_,
  #  lr = deeptrees_raw$`lr-Adam`[idx]
  #)
#}

# Baue DeepTrees aus allen Epochenzeilen
#deeptrees <- map_dfr(epoch_rows, extract_epoch_block) %>%
  #filter(!is.na(train_loss) & !is.na(val_loss)) %>%  # Nur vollständige Epochen
  #mutate(method = "DeepTrees") %>%
  #arrange(epoch)

#cat("✓ DeepTrees geladen:", nrow(deeptrees), "Epochen\n")
############


# ─────────────────────────────────────────────────────────────────────────
# COMBINE ALL METHODS
# ─────────────────────────────────────────────────────────────────────────

all_metrics <- bind_rows(tcd, detectree2, deeptrees) %>%
  mutate(method = factor(method, levels = c("TCD", "Detectree2", "DeepTrees")))

colors <- c("TCD" = "#E74C3C", "Detectree2" = "#3498DB", "DeepTrees" = "#2ECC71")

cat("✓ Daten kombiniert:\n")
cat(sprintf("  - TCD: %d Epochen\n", n_distinct(tcd$epoch)))
cat(sprintf("  - Detectree2: %d Epochen\n", n_distinct(detectree2$epoch)))
cat(sprintf("  - DeepTrees: %d Epochen\n", n_distinct(deeptrees$epoch)))

# ═══════════════════════════════════════════════════════════════════════════
# 2. SMOOTHED LOSS CURVES
# ═══════════════════════════════════════════════════════════════════════════

cat("\n📊 Erstelle Smoothed Loss Curves...\n")

all_metrics_smooth <- all_metrics %>%
  group_by(method) %>%
  arrange(epoch) %>%
  mutate(
    train_loss_smooth = zoo::rollmean(train_loss, k = min(5, n()), fill = NA, align = "center", na.rm = TRUE),
    val_loss_smooth = zoo::rollmean(val_loss, k = min(5, n()), fill = NA, align = "center", na.rm = TRUE)
  ) %>%
  ungroup()

p_smooth_train <- ggplot(all_metrics_smooth, aes(x = epoch, color = method)) +
  geom_line(aes(y = train_loss), alpha = 0.5, linewidth = 0.5) +
  geom_line(aes(y = train_loss_smooth), linewidth = 1.5) +
  scale_color_manual(values = colors) +
  labs(title = "Smoothed Training Loss (5-epoch moving average)",
       x = "Epoch", y = "Training Loss", color = "Method") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))

p_smooth_val <- ggplot(all_metrics_smooth, aes(x = epoch, color = method)) +
  geom_line(aes(y = val_loss), alpha = 0.5, linewidth = 0.5) +
  geom_line(aes(y = val_loss_smooth), linewidth = 1.5) +
  scale_color_manual(values = colors) +
  labs(title = "Smoothed Validation Loss (5-epoch moving average)",
       x = "Epoch", y = "Validation Loss", color = "Method") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))

p_smooth_combined <- p_smooth_train / p_smooth_val +
  plot_annotation(title = "Smoothed Loss Curves - Trend Analysis",
                  theme = theme(plot.title = element_text(size = 18, face = "bold")))

ggsave(file.path(output_dir, "03_smoothed_loss_curves.png"),
       p_smooth_combined, width = 12, height = 10, dpi = 300)

cat("✓ Smoothed Loss Curves gespeichert\n")

# ═══════════════════════════════════════════════════════════════════════════
# 3. CONVERGENCE SPEED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

cat("\n📊 Analysiere Convergence Speed...\n")

convergence_analysis <- all_metrics %>%
  group_by(method) %>%
  filter(n() > 1) %>%
  arrange(epoch) %>%
  mutate(
    train_loss_improvement = lag(train_loss) - train_loss,
    val_loss_improvement = lag(val_loss) - val_loss
  ) %>%
  filter(!is.na(train_loss_improvement)) %>%
  summarise(
    avg_train_improvement = mean(train_loss_improvement, na.rm = TRUE),
    avg_val_improvement = mean(val_loss_improvement, na.rm = TRUE),
    epochs_to_80pct_best = {
      best_val <- min(val_loss, na.rm = TRUE)
      first_80pct <- which(val_loss <= best_val * 1.2)[1]
      ifelse(is.na(first_80pct), NA, first_80pct)
    },
    .groups = "drop"
  )

cat("\n📋 CONVERGENCE SPEED SUMMARY:\n")
print(convergence_analysis)

write_csv(convergence_analysis, file.path(output_dir, "convergence_speed.csv"))

# Plot: Improvement Rate
p_improvement <- all_metrics %>%
  group_by(method) %>%
  filter(n() > 5) %>%
  arrange(epoch) %>%
  mutate(train_loss_improvement = lag(train_loss) - train_loss) %>%
  filter(!is.na(train_loss_improvement), epoch > 5) %>%
  ggplot(aes(x = epoch, y = train_loss_improvement, color = method)) +
  geom_line(linewidth = 1.2, alpha = 0.7) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
  scale_color_manual(values = colors) +
  labs(title = "Loss Improvement Rate per Epoch",
       subtitle = "Positive values = loss decreased",
       x = "Epoch", y = "Loss Improvement", color = "Method") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "04_convergence_speed.png"),
       p_improvement, width = 12, height = 6, dpi = 300)

cat("✓ Convergence Speed Plot gespeichert\n")

# ═══════════════════════════════════════════════════════════════════════════
# 4. OVERFITTING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

cat("\n📊 Analysiere Overfitting (Train/Val Gap)...\n")

overfitting_data <- all_metrics %>%
  mutate(loss_gap = train_loss - val_loss,
         gap_percent = (train_loss - val_loss) / val_loss * 100)

p_overfitting <- ggplot(overfitting_data, aes(x = epoch, y = loss_gap, color = method)) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = colors) +
  labs(title = "Overfitting Analysis: Train-Val Loss Gap",
       subtitle = "Negative = Validation loss higher (potential overfitting)",
       x = "Epoch", y = "Train Loss - Val Loss", color = "Method") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "05_overfitting_analysis.png"),
       p_overfitting, width = 12, height = 6, dpi = 300)

cat("✓ Overfitting Analysis Plot gespeichert\n")

# ═══════════════════════════════════════════════════════════════════════════
# 5. LEARNING RATE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

cat("\n📊 Erstelle Learning Rate Plots...\n")

if(any(!is.na(all_metrics$lr))) {
  p_lr <- ggplot(all_metrics %>% filter(!is.na(lr)),
                 aes(x = epoch, y = lr, color = method)) +
    geom_line(linewidth = 1.2) +
    scale_color_manual(values = colors) +
    scale_y_log10() +
    labs(title = "Learning Rate Schedule Comparison",
         x = "Epoch", y = "Learning Rate (log scale)", color = "Method") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "bottom")
  
  ggsave(file.path(output_dir, "06_learning_rate_schedule.png"),
         p_lr, width = 12, height = 6, dpi = 300)
  
  cat("✓ Learning Rate Schedule Plot gespeichert\n")
}

# ═══════════════════════════════════════════════════════════════════════════
# 6. DISTRIBUTION ANALYSIS (Final 10 Epochs)
# ═══════════════════════════════════════════════════════════════════════════

cat("\n📊 Erstelle Distribution Plots (Final 10 Epochs)...\n")

final_10_epochs <- all_metrics %>%
  group_by(method) %>%
  arrange(desc(epoch)) %>%
  slice_head(n = 10) %>%
  ungroup()

p_boxplot_train <- ggplot(final_10_epochs, aes(x = method, y = train_loss, fill = method)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.5) +
  scale_fill_manual(values = colors) +
  labs(title = "Training Loss Distribution (Final 10 Epochs)",
       x = "Method", y = "Training Loss") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

p_boxplot_val <- ggplot(final_10_epochs, aes(x = method, y = val_loss, fill = method)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.5) +
  scale_fill_manual(values = colors) +
  labs(title = "Validation Loss Distribution (Final 10 Epochs)",
       x = "Method", y = "Validation Loss") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

p_distributions <- p_boxplot_train | p_boxplot_val

ggsave(file.path(output_dir, "07_loss_distributions.png"),
       p_distributions, width = 12, height = 6, dpi = 300)

cat("✓ Distribution Plots gespeichert\n")

# ═══════════════════════════════════════════════════════════════════════════
# 7. STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

cat("\n📊 Führe statistische Tests durch...\n")

methods <- unique(final_10_epochs$method)

if(length(methods) >= 2) {
  pairs <- combn(methods, 2, simplify = FALSE)
  
  for(pair in pairs) {
    m1 <- pair[1]
    m2 <- pair[2]
    
    vals1 <- final_10_epochs %>% filter(method == m1) %>% pull(val_loss)
    vals2 <- final_10_epochs %>% filter(method == m2) %>% pull(val_loss)
    
    if(length(vals1) > 0 && length(vals2) > 0 && !all(is.na(vals1)) && !all(is.na(vals2))) {
      wilcox_result <- wilcox.test(vals1, vals2, na.action = na.omit)
      t_result <- t.test(vals1, vals2, na.action = na.omit)
      
      cat(sprintf("\n🔬 TESTS: %s vs %s (Final 10 Epochs - Val Loss)\n", m1, m2))
      cat(sprintf("  Valid samples: %d vs %d\n", sum(!is.na(vals1)), sum(!is.na(vals2))))
      cat(sprintf("  Wilcoxon p-value: %.6f (Sig: %s)\n",
                  wilcox_result$p.value,
                  ifelse(wilcox_result$p.value < 0.05, "YES", "NO")))
      cat(sprintf("  T-Test p-value: %.6f\n", t_result$p.value))
      cat(sprintf("  %s mean: %.4f ± %.4f\n", m1, mean(vals1, na.rm = TRUE), sd(vals1, na.rm = TRUE)))
      cat(sprintf("  %s mean: %.4f ± %.4f\n", m2, mean(vals2, na.rm = TRUE), sd(vals2, na.rm = TRUE)))
    }
  }
}

# ═══════════════════════════════════════════════════════════════════════════
# 8. COMPREHENSIVE SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

final_summary <- all_metrics %>%
  group_by(method) %>%
  summarise(
    Final_Epoch = max(epoch),
    Final_Train_Loss = last(train_loss[!is.na(train_loss)]),
    Final_Val_Loss = last(val_loss[!is.na(val_loss)]),
    Best_Val_Loss = min(val_loss, na.rm = TRUE),
    Best_Epoch = epoch[which.min(val_loss)],
    Final_Train_IoU = last(train_iou[!is.na(train_iou)]),
    Final_Val_IoU = last(val_iou[!is.na(val_iou)]),
    Avg_Loss_Improvement = mean(c(lag(val_loss) - val_loss)[-1], na.rm = TRUE),
    .groups = "drop"
  )

cat("\n📋 COMPREHENSIVE SUMMARY:\n")
print(final_summary %>% mutate(across(where(is.numeric), ~round(., 4))))

write_csv(final_summary, file.path(output_dir, "final_summary.csv"))

# ═══════════════════════════════════════════════════════════════════════════
# FERTIG!
# ═══════════════════════════════════════════════════════════════════════════

cat("\n╔═══════════════════════════════════════════════════════════════╗\n")
cat("║ ✅ ERWEITERTE ANALYSE ABGESCHLOSSEN!                         ║\n")
cat("║                                                               ║\n")
cat("║ 📊 Generierte Analysen:                                       ║\n")
cat("║ • 03_smoothed_loss_curves.png                                ║\n")
cat("║ • 04_convergence_speed.png                                   ║\n")
cat("║ • 05_overfitting_analysis.png                                ║\n")
cat("║ • 06_learning_rate_schedule.png                              ║\n")
cat("║ • 07_loss_distributions.png                                  ║\n")
cat("║ • convergence_speed.csv                                      ║\n")
cat("║ • final_summary.csv                                          ║\n")
cat("║                                                               ║\n")
cat("║ 📍 Output Directory:                                          ║\n")
cat(sprintf("║ %s\n", output_dir))
cat("║                                                               ║\n")
cat("║ ✨ DeepTrees erfolgreich integriert!                         ║\n")
cat("╚═══════════════════════════════════════════════════════════════╝\n\n")



#str(detectree2_metrics_path)
#summary(detectree2_metrics_path$iteration)
#summary(detectree2_metrics_path$total_loss)
#sum(is.na(detectree2_metrics_path$total_loss))

