library(tidyverse)
library(reshape2)

# Daten laden
df <- read.csv("/home/abrax/Desktop/Bitz_singular/output/multimethod/pipe2_statistics/pipe2_correlations.csv")
# long format: cor_height/cor_dbh/cor_crown -> eine Spalte
df_long <- df %>%
  select(method, starts_with("cor_")) %>%
  pivot_longer(
    cols = starts_with("cor_"),
    names_to = "metric",
    values_to = "correlation"
  ) %>%
  mutate(
    metric = recode(metric,
                    cor_height = "Height (CHM vs GT)",
                    cor_dbh    = "DBH (pred vs GT)",
                    cor_crown  = "Crown area (pred vs GT)")
  )

ggplot(df_long, aes(x = metric, y = method, fill = correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", correlation)), size = 4) +
  scale_fill_gradient2(midpoint = 0, limits = c(-1, 1), name = "r") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    panel.grid = element_blank()
  ) +
  labs(x = NULL, y = NULL)

#####



df_long <- df %>%
  select(method, starts_with("cor_")) %>%
  pivot_longer(cols = starts_with("cor_"),
               names_to = "metric",
               values_to = "correlation")

ggplot(df_long, aes(x = metric, y = correlation, fill = method)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  theme_minimal() +
  labs(x = NULL, y = "Correlation (r)", fill = "Method")