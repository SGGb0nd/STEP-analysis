#!/usr/bin/env Rscript
# Plot singular-point distances for the CRC tumor-periphery continuity check.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
input_csv <- if (length(args) >= 1) args[[1]] else "periphery_distances.csv"
out_dir <- if (length(args) >= 2) args[[2]] else "workflows/crc_periphery_continuity"

grid_step_um <- 8
display_max_um <- 80
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read_csv(input_csv, show_col_types = FALSE) %>%
  mutate(
    distance = as.numeric(distance),
    distance_um = distance * grid_step_um,
    method = if_else(method == paste0("50", intToUtf8(181), "m"), "50 um", method),
    batch = recode(batch, cancer_p1 = "cancer p1", cancer_p2 = "cancer p2", cancer_p5 = "cancer p5"),
    method = factor(method, levels = c("50 um", "STEP")),
    batch = factor(batch, levels = c("cancer p1", "cancer p2", "cancer p5"))
  )

df_filtered <- df %>%
  filter(distance_um <= display_max_um) %>%
  group_by(method, batch) %>%
  mutate(width_scale = n() / max(table(interaction(df$method, df$batch)))) %>%
  ungroup()

y_min_um <- df_filtered %>%
  filter(distance_um > 0) %>%
  summarise(min_distance_um = min(distance_um, na.rm = TRUE)) %>%
  pull(min_distance_um)

summary_df <- df %>%
  group_by(batch, method) %>%
  summarise(
    total_singular_points = n(),
    shown_points = sum(distance_um <= display_max_um),
    shown_pct = 100 * shown_points / total_singular_points,
    .groups = "drop"
  )

p <- ggplot(df_filtered, aes(x = method, y = distance_um)) +
  geom_violin(aes(width = width_scale), alpha = 0.3, fill = "lightblue", color = "black") +
  geom_jitter(alpha = 0.35, color = "red", size = 0.45, width = 0.16) +
  facet_wrap(~batch, nrow = 1) +
  theme_bw(base_size = 16) +
  theme(
    axis.title.x = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
    strip.text = element_text(face = "bold")
  ) +
  labs(y = expression("Singularity distance (" * mu * "m)"), x = NULL) +
  coord_cartesian(ylim = c(y_min_um, display_max_um))

ggsave(file.path(out_dir, "crc_singularity_distance_distribution_um.pdf"), p, width = 6, height = 4)
ggsave(file.path(out_dir, "crc_singularity_distance_distribution_um.png"), p, width = 6, height = 4, dpi = 300)
write_csv(summary_df, file.path(out_dir, "summary.csv"))
