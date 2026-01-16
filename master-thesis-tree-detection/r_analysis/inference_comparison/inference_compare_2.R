# ═════════════════════════════════════════════════════════════════════════════
# 📊 INFERENCE COMPARISON – STATISTICAL & GRAPHICAL ANALYSIS
# • Vergleich TCD vs. Detectree2 vs. DeepTree
# • Getrennte Auswertung für Train vs. Test GT
# • Umfassende Statistik + hochwertige Visualisierungen
# • Basiert auf pipe1_summary_all_methods.csv (mit gt_set Spalte)
# ═════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(gridExtra)
  library(scales)
  library(ggpubr)
  library(RColorBrewer)
})

# ═══ 1. KONFIGURATION ═══

input_csv <- "/home/abrax/Desktop/Infer_stat_input/inference_comparison/pipe1_summary_all_methods.csv"
output_dir <- "/home/abrax/Desktop/Infer_stat_input/inference_comparison/statistics"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ═══ 2. DATEN LADEN ═══

df <- read_csv(input_csv, show_col_types = FALSE)

cat("✓ Daten geladen:\n")
print(df)

# Datenvalidierung
if (nrow(df) == 0) {
  stop("❌ CSV-Datei ist leer! Bitte prüfe den Pfad und Dateiinhalt.")
}

# Stelle sicher, dass gt_set Spalte vorhanden ist
if (!"gt_set" %in% names(df)) {
  cat("⚠️  'gt_set'-Spalte nicht gefunden – setze standardmäßig auf 'train'\n")
  df$gt_set <- "train"
}

cat(sprintf("\n✓ Daten: %d Zeilen, %d Spalten\n", nrow(df), ncol(df)))
cat(sprintf("  Train-Metriken: %d\n", sum(df$gt_set == "train", na.rm = TRUE)))
cat(sprintf("  Test-Metriken: %d\n", sum(df$gt_set == "test", na.rm = TRUE)))

# ═══ 3. DESKRIPTIVE STATISTIK ═══

cat("\n════════════════════════════════════════\n")
cat("📊 DESKRIPTIVE STATISTIK\n")
cat("════════════════════════════════════════\n\n")

# 3.1 Pro Methode aggregiert (ohne gt_set Aufteilung)
stats_method <- df %>%
  group_by(method) %>%
  summarise(
    n_tiles = n(),
    mean_Recall = round(mean(Recall, na.rm = TRUE), 3),
    sd_Recall = round(sd(Recall, na.rm = TRUE), 3),
    mean_Precision = round(mean(Precision, na.rm = TRUE), 3),
    sd_Precision = round(sd(Precision, na.rm = TRUE), 3),
    mean_F1 = round(mean(F1, na.rm = TRUE), 3),
    sd_F1 = round(sd(F1, na.rm = TRUE), 3),
    mean_IoU = round(mean(mean_IoU, na.rm = TRUE), 3),
    sd_IoU = round(sd(mean_IoU, na.rm = TRUE), 3),
    total_TP = sum(TP, na.rm = TRUE),
    total_FP = sum(FP, na.rm = TRUE),
    total_FN = sum(FN, na.rm = TRUE),
    .groups = "drop"
  )

cat("👉 Statistik pro Methode (insgesamt):\n")
print(stats_method)
write_csv(stats_method, file.path(output_dir, "statistics_by_method.csv"))

# 3.2 Pro Methode UND gt_set (Train/Test getrennt)
stats_method_set <- df %>%
  group_by(method, gt_set) %>%
  summarise(
    n_tiles = n(),
    mean_Recall = round(mean(Recall, na.rm = TRUE), 3),
    sd_Recall = round(sd(Recall, na.rm = TRUE), 3),
    mean_Precision = round(mean(Precision, na.rm = TRUE), 3),
    sd_Precision = round(sd(Precision, na.rm = TRUE), 3),
    mean_F1 = round(mean(F1, na.rm = TRUE), 3),
    sd_F1 = round(sd(F1, na.rm = TRUE), 3),
    mean_IoU = round(mean(mean_IoU, na.rm = TRUE), 3),
    sd_IoU = round(sd(mean_IoU, na.rm = TRUE), 3),
    total_TP = sum(TP, na.rm = TRUE),
    total_FP = sum(FP, na.rm = TRUE),
    total_FN = sum(FN, na.rm = TRUE),
    .groups = "drop"
  )

cat("\n👉 Statistik pro Methode UND gt_set (Train vs. Test):\n")
print(stats_method_set)
write_csv(stats_method_set, file.path(output_dir, "statistics_by_method_and_set.csv"))

# 3.3 Pro Tile (Species) aggregiert
stats_tile <- df %>%
  group_by(tile, species) %>%
  summarise(
    n_methods = n(),
    mean_Recall = round(mean(Recall, na.rm = TRUE), 3),
    mean_Precision = round(mean(Precision, na.rm = TRUE), 3),
    mean_F1 = round(mean(F1, na.rm = TRUE), 3),
    mean_IoU = round(mean(mean_IoU, na.rm = TRUE), 3),
    .groups = "drop"
  )

cat("\n👉 Statistik pro Tile (Species):\n")
print(stats_tile)
write_csv(stats_tile, file.path(output_dir, "statistics_by_tile.csv"))

# 3.4 Pro Tile UND gt_set
stats_tile_set <- df %>%
  group_by(tile, species, gt_set) %>%
  summarise(
    n_methods = n(),
    mean_Recall = round(mean(Recall, na.rm = TRUE), 3),
    mean_Precision = round(mean(Precision, na.rm = TRUE), 3),
    mean_F1 = round(mean(F1, na.rm = TRUE), 3),
    mean_IoU = round(mean(mean_IoU, na.rm = TRUE), 3),
    .groups = "drop"
  )

cat("\n👉 Statistik pro Tile UND gt_set:\n")
print(stats_tile_set)
write_csv(stats_tile_set, file.path(output_dir, "statistics_by_tile_and_set.csv"))

# ═══ 4. VISUALISIERUNGEN ═══

cat("\n════════════════════════════════════════\n")
cat("📈 VISUALISIERUNGEN\n")
cat("════════════════════════════════════════\n\n")

# Farbpalette für Methoden
method_colors <- c("TCD" = "#E69F00", "Detectree2" = "#56B4E9", "DeepTree" = "#009E73")

# ──────── 4.1 | Precision, Recall, F1 – Balkendiagramm (mit gt_set Facet) ────────

p1 <- df %>%
  select(method, species, gt_set, Recall, Precision, F1) %>%
  pivot_longer(cols = c(Recall, Precision, F1), names_to = "Metric", values_to = "Value") %>%
  ggplot(aes(x = species, y = Value, fill = method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  facet_grid(gt_set ~ Metric, scales = "free_y") +
  scale_fill_manual(values = method_colors) +
  labs(
    title = "Detection Performance by Species, Method & GT-Set",
    subtitle = "Precision, Recall, F1-Score – Train vs Test",
    x = "Species (Tile)",
    y = "Score",
    fill = "Method"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 16),
    strip.text = element_text(face = "bold", size = 11),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

ggsave(file.path(output_dir, "01_metrics_by_species_and_set.png"), p1, width = 14, height = 10, dpi = 300)
cat("✓ Grafik gespeichert: 01_metrics_by_species_and_set.png\n")

# ──────── 4.2 | mean IoU – Vergleich (mit gt_set) ────────

p2 <- ggplot(df, aes(x = species, y = mean_IoU, fill = method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ gt_set, ncol = 2) +
  scale_fill_manual(values = method_colors) +
  labs(
    title = "Mean IoU Comparison by Species, Method & GT-Set",
    subtitle = "Average Intersection over Union for True Positives",
    x = "Species (Tile)",
    y = "Mean IoU",
    fill = "Method"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 16),
    strip.text = element_text(face = "bold", size = 11)
  )

ggsave(file.path(output_dir, "02_mean_iou_comparison_by_set.png"), p2, width = 12, height = 6, dpi = 300)
cat("✓ Grafik gespeichert: 02_mean_iou_comparison_by_set.png\n")

# ──────── 4.3 | TP/FP/FN – Stacked Bar (mit gt_set) ────────

df_long_tpfp <- df %>%
  select(method, species, gt_set, TP, FP, FN) %>%
  pivot_longer(cols = c(TP, FP, FN), names_to = "Category", values_to = "Count")

p3 <- ggplot(df_long_tpfp, aes(x = species, y = Count, fill = Category)) +
  geom_bar(stat = "identity", position = "stack") +
  facet_grid(gt_set ~ method) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "True Positives, False Positives, False Negatives",
    subtitle = "Detection outcomes by method, species & GT-Set (train vs test)",
    x = "Species (Tile)",
    y = "Count",
    fill = "Category"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 16),
    strip.text = element_text(face = "bold", size = 10)
  )

ggsave(file.path(output_dir, "03_tp_fp_fn_stacked_by_set.png"), p3, width = 14, height = 8, dpi = 300)
cat("✓ Grafik gespeichert: 03_tp_fp_fn_stacked_by_set.png\n")

# ──────── 4.4 | Recall vs. Precision – Scatterplot (mit gt_set) ────────

p4 <- ggplot(df, aes(x = Recall, y = Precision, color = method, shape = species, size = F1)) +
  geom_point(alpha = 0.8) +
  facet_wrap(~ gt_set, ncol = 2) +
  scale_color_manual(values = method_colors) +
  scale_size_continuous(range = c(3, 8)) +
  labs(
    title = "Recall vs. Precision by GT-Set (Train vs Test)",
    subtitle = "Trade-off between detection sensitivity and accuracy; size = F1-Score",
    x = "Recall (Sensitivity)",
    y = "Precision (Accuracy)",
    color = "Method",
    shape = "Species",
    size = "F1-Score"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", size = 16),
    strip.text = element_text(face = "bold", size = 11)
  ) +
  coord_cartesian(xlim = c(0.7, 1.0), ylim = c(0.5, 1.0))

ggsave(file.path(output_dir, "04_recall_vs_precision_by_set.png"), p4, width = 12, height = 6, dpi = 300)
cat("✓ Grafik gespeichert: 04_recall_vs_precision_by_set.png\n")

# ──────── 4.5 | Boxplot – Verteilung der Metriken (mit gt_set) ────────

df_metrics_long <- df %>%
  select(method, gt_set, Recall, Precision, F1) %>%
  pivot_longer(cols = c(Recall, Precision, F1), names_to = "Metric", values_to = "Value")

p5 <- ggplot(df_metrics_long, aes(x = method, y = Value, fill = method)) +
  geom_boxplot(alpha = 0.7) +
  facet_grid(gt_set ~ Metric, scales = "free_y") +
  scale_fill_manual(values = method_colors) +
  labs(
    title = "Distribution of Detection Metrics by GT-Set",
    subtitle = "Boxplots for Recall, Precision, F1-Score; Train vs Test",
    x = "Method",
    y = "Score",
    fill = "Method"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 16),
    strip.text = element_text(face = "bold", size = 11)
  )

ggsave(file.path(output_dir, "05_metric_distributions_by_set.png"), p5, width = 14, height = 8, dpi = 300)
cat("✓ Grafik gespeichert: 05_metric_distributions_by_set.png\n")

# ──────── 4.6 | Heatmap – F1-Score Methode × Species × gt_set ────────

# Separate Heatmaps für Train und Test
for (set_name in c("train", "test")) {
  df_set <- df %>% filter(gt_set == set_name)
  
  p_heatmap <- ggplot(df_set, aes(x = species, y = method, fill = F1)) +
    geom_tile(color = "white", size = 1) +
    geom_text(aes(label = round(F1, 2)), color = "black", size = 5, fontface = "bold") +
    scale_fill_gradient(low = "lightcoral", high = "darkgreen", limits = c(0.5, 1.0), na.value = "grey") +
    labs(
      title = sprintf("F1-Score Heatmap – GT-Set: %s", toupper(set_name)),
      subtitle = "Method × Species performance",
      x = "Species (Tile)",
      y = "Method",
      fill = "F1-Score"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.text.x = element_text(angle = 0, hjust = 0.5)
    )
  
  file_name <- sprintf("06_f1_heatmap_%s.png", set_name)
  ggsave(file.path(output_dir, file_name), p_heatmap, width = 8, height = 5, dpi = 300)
  cat(sprintf("✓ Grafik gespeichert: %s\n", file_name))
}

# ──────── 4.7 | Fehleranalyse: FP & FN (mit gt_set) ────────

p7 <- df %>%
  select(method, species, gt_set, FP, FN) %>%
  pivot_longer(cols = c(FP, FN), names_to = "Error_Type", values_to = "Count") %>%
  ggplot(aes(x = species, y = Count, fill = Error_Type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  facet_grid(gt_set ~ method) +
  scale_fill_manual(values = c("FP" = "#D55E00", "FN" = "#CC79A7")) +
  labs(
    title = "False Positives vs. False Negatives by GT-Set",
    subtitle = "Error distribution by method, species, train vs test",
    x = "Species (Tile)",
    y = "Count",
    fill = "Error Type"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 16),
    strip.text = element_text(face = "bold", size = 10)
  )

ggsave(file.path(output_dir, "07_fp_fn_comparison_by_set.png"), p7, width = 14, height = 8, dpi = 300)
cat("✓ Grafik gespeichert: 07_fp_fn_comparison_by_set.png\n")

# ═══ 5. STATISTISCHE TESTS ═══

cat("\n════════════════════════════════════════\n")
cat("📈 STATISTISCHE TESTS\n")
cat("════════════════════════════════════════\n\n")

# Wilcoxon-Test: TCD vs. Detectree2 (für Train- und Test-Set separat)
if ("TCD" %in% df$method & "Detectree2" %in% df$method) {
  for (set_name in c("train", "test")) {
    df_set <- df %>% filter(gt_set == set_name)
    tcd_recall <- df_set %>% filter(method == "TCD") %>% pull(Recall)
    dt2_recall <- df_set %>% filter(method == "Detectree2") %>% pull(Recall)
    
    if (length(tcd_recall) > 0 && length(dt2_recall) > 0) {
      test_result <- wilcox.test(tcd_recall, dt2_recall)
      cat(sprintf("\n🔬 Wilcoxon Test – Recall: TCD vs. Detectree2 (%s)\n", toupper(set_name)))
      cat(sprintf("  p-value: %.6f\n", test_result$p.value))
      cat(sprintf("  Significant (α=0.05): %s\n", ifelse(test_result$p.value < 0.05, "YES", "NO")))
    }
  }
}

# ═══ 6. ZUSAMMENFASSUNG ═══

cat("\n════════════════════════════════════════\n")
cat("✅ ANALYSE ABGESCHLOSSEN\n")
cat("════════════════════════════════════════\n")
cat(sprintf("📂 Output-Ordner: %s\n\n", output_dir))

cat("📄 Erzeugte CSV-Dateien:\n")
cat(" • statistics_by_method.csv (insgesamt)\n")
cat(" • statistics_by_method_and_set.csv (Train vs Test)\n")
cat(" • statistics_by_tile.csv (nach Species)\n")
cat(" • statistics_by_tile_and_set.csv (nach Species & Train/Test)\n\n")

cat("📊 Erzeugte Grafiken:\n")
cat(" • 01_metrics_by_species_and_set.png (Precision/Recall/F1 – facettiert nach gt_set)\n")
cat(" • 02_mean_iou_comparison_by_set.png (IoU – facettiert nach gt_set)\n")
cat(" • 03_tp_fp_fn_stacked_by_set.png (TP/FP/FN gestapelt – facettiert nach gt_set×Methode)\n")
cat(" • 04_recall_vs_precision_by_set.png (Scatterplot – facettiert nach gt_set)\n")
cat(" • 05_metric_distributions_by_set.png (Boxplots – facettiert nach gt_set×Metrik)\n")
cat(" • 06_f1_heatmap_train.png (F1-Score Heatmap nur Train)\n")
cat(" • 06_f1_heatmap_test.png (F1-Score Heatmap nur Test)\n")
cat(" • 07_fp_fn_comparison_by_set.png (FP vs FN – facettiert nach gt_set×Methode)\n\n")

cat("📈 Interpretationshilfe:\n")
cat(" • Train-Set (85%): Performance auf Trainingsdaten (Hinweis auf Overfitting/Trainingsqualität)\n")
cat(" • Test-Set (15%): Performance auf Holdout-Daten (echte Generalisierungsfähigkeit)\n")
cat(" • Differenz Train→Test: Je größer, desto stärker Overfitting\n")