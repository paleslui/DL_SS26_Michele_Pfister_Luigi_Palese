#install.packages("devtools")
#devtools::install_github("GfellerLab/EPIC", build_vignettes = TRUE)
library(EPIC)
data("TRef", package = "EPIC")

bulk <- read.csv("/Users/michelepfister/Documents/Tracking_module/ucec_tpm.csv", 
                   header = TRUE, 
                   row.names = 1, 
                   sep = ",",           # Changed to comma separator
                   quote = "\"",        # Handle quoted strings
                   check.names = FALSE)

# Convert to matrix (EPIC expects a matrix)
bulk_matrix <- as.matrix(bulk)

# Run EPIC
results <- EPIC(bulk = bulk_matrix, reference = TRef, scaleExprs = TRUE)

table(results$fit.gof$convergeCode)

# View results
print(results$cellFractions)

write.csv(results$cellFractions, "ucec_epic_cellFractions.csv")

# Load EPIC results and clinical data
epic <- read.csv("ucec_epic_cellFractions.csv", row.names = 1)
clinical <- read.csv("/Users/michelepfister/Documents/Tracking_module/ucec_clinical.csv")

# Extract patient ID from sample names (TCGA format)
epic$patient_id <- sub("^((TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})).*", "\\1", rownames(epic))

# Move rownames to a column named sample_submitter_id
epic$sample_submitter_id <- rownames(epic)

# Reorder columns to put sample_submitter_id first
epic <- epic[, c("sample_submitter_id", setdiff(names(epic), "sample_submitter_id"))]

# Save with proper column name
write.csv(epic, "ucec_epic_cellFractions.csv", row.names = FALSE)


# Merge the two datasets
merged_data <- merge(clinical, epic, by = "sample_submitter_id", all = FALSE)

# Save merged data
write.csv(merged_data, "ucec_merged_clinical_epic.csv", row.names = FALSE)

# View result
head(merged_data)
dim(merged_data)

# clean the merged data
merged <- read.csv("ucec_merged_clinical_epic.csv")

# Remove rows with empty strings in MSI column
merged_clean <- merged[merged$MSI != "", ]

# Check cleaned data
cat("\nCleaned data dimensions:", dim(merged_clean), "\n")
cat("MSI column after cleaning:\n")
print(table(merged_clean$MSI, useNA = "always"))

# View result
dim(merged_clean)

# Load required libraries
library(ggplot2)
library(reshape2)
library(dplyr)
library(tidyr)
library(ggpubr)

#On average (median)
aggregate(CD8_Tcells ~ MSI, data = merged_clean, median)
aggregate(Macrophages ~ MSI, data = merged_clean, median)
aggregate(NKcells ~ MSI, data = merged_clean, median)
aggregate(Bcells ~ MSI, data = merged_clean, median)
aggregate(CD4_Tcells ~ MSI, data = merged_clean, median)
aggregate(otherCells ~ MSI, data = merged_clean, median)
aggregate(CAFs ~ MSI, data = merged_clean, median)
aggregate(Endothelial ~ MSI, data = merged_clean, median)

# Identify immune cell columns (from EPIC output)
immune_cells <- c("Bcells", "CD4_Tcells", "CD8_Tcells", "Macrophages", "NKcells")

# Statistical comparison (t-tests)
cat("\n=== Statistical Comparison (MSS vs MSI-H) ===\n")
comparison_results <- data.frame(
  Cell_Type = character(),
  MSS_mean = numeric(),
  MSI_mean = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

for(cell in immune_cells) {
  mss_values <- merged_clean[merged_clean$MSI == "MSS", cell]
  msi_values <- merged_clean[merged_clean$MSI == "MSI-H", cell]
  
  test_result <- t.test(mss_values, msi_values)
  
  comparison_results <- rbind(comparison_results, data.frame(
    Cell_Type = cell,
    MSS_mean = mean(mss_values, na.rm = TRUE),
    MSI_mean = mean(msi_values, na.rm = TRUE),
    p_value = test_result$p.value
  ))
}

print(comparison_results)

# Create boxplots for each cell type
# Reshape data for plotting
plot_data <- merged_clean[, c("MSI", immune_cells)]
plot_data_long <- melt(plot_data, id.vars = "MSI", 
                       variable.name = "Cell_Type", 
                       value.name = "Fraction")


# Create faceted boxplot
p <- ggplot(plot_data_long, aes(x = MSI, y = Fraction, fill = MSI)) +
  geom_boxplot(width = 0.6, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 0.7, alpha = 0.35) +
  facet_wrap(~ Cell_Type, scales = "free_y") +
  theme_bw(base_size = 12) +
  theme(
    strip.background = element_rect(fill = "grey90", color = NA),
    strip.text = element_text(face = "bold"),
    legend.position = "none"
  ) +
  labs(
    title = "Immune Cell Fractions by MSI Status (TCGA-UCEC)",
    x = "MSI Status",
    y = "Cell Fraction"
  ) +
  scale_fill_manual(values = c("MSS" = "#E69F00", "MSI-H" = "#56B4E9"))

ggsave("immune_cells_boxplot_facet.pdf", p, width = 12, height = 7)
print(p)




# Identify stromal cell columns (from EPIC output)
stromal_candidates <- c("CAFs", "Endothelial", "OtherCells", "otherCells", "Other")

# only the ones that actually exist in the data
stromal_present <- intersect(stromal_candidates, colnames(merged_clean))

# stop with a clear message if none found
if (length(stromal_present) == 0) {
  stop("No stromal columns found. Check EPIC output column names in merged_clean.")
}

plot_stromal_long <- merged_clean %>%
  select(MSI, all_of(stromal_present)) %>%
  pivot_longer(
    cols = all_of(stromal_present),
    names_to = "Cell_Type",
    values_to = "Fraction"
  ) %>%
  mutate(
    MSI = factor(MSI, levels = c("MSS", "MSI-H")),
    Cell_Type = factor(Cell_Type, levels = stromal_present)
  )

p_stromal <- ggplot(plot_stromal_long, aes(x = MSI, y = Fraction, fill = MSI)) +
  geom_boxplot(width = 0.6, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 0.8, alpha = 0.35) +
  facet_wrap(~ Cell_Type, scales = "free_y", ncol = 3) +
  theme_bw(base_size = 12) +
  theme(
    strip.background = element_rect(fill = "grey90", color = NA),
    strip.text = element_text(face = "bold"),
    legend.position = "none"
  ) +
  labs(
    title = "Stromal / Other Cell Fractions by MSI Status (TCGA-UCEC)",
    x = "MSI Status",
    y = "Cell Fraction"
  ) +
  scale_fill_manual(values = c("MSS" = "#E69F00", "MSI-H" = "#56B4E9"))

ggsave("stromal_cells_boxplot_facet.pdf", p_stromal, width = 10, height = 5)
print(p_stromal)

# Wilcoxon tests for CD8 T cells and Macrophages
wilcox.test(CD8_Tcells ~ MSI, data = merged_clean)
wilcox.test(Macrophages ~ MSI, data = merged_clean)

