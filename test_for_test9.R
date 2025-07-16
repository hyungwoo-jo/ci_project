# -------------------------------------------------------------------
# 0. Package Installation (Run only once)
# -------------------------------------------------------------------
# install.packages(c("tidyverse", "MatchIt", "cobalt", "DoubleML", "mlr3", "mlr3learners", "ranger", "future", "progress", "grf", "bartCause", "mlr3extralearners"))
install.packages("remotes")
remotes::install_github("mlr-org/mlr3extralearners@*release")

# -------------------------------------------------------------------
# 1. Load Packages
# -------------------------------------------------------------------
library(tidyverse)
library(MatchIt)
library(cobalt)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(ranger)
library(future)
library(progress)
library(grf)
library(bartCause)
library(mlr3extralearners)

# -------------------------------------------------------------------
# 2. Parallel Processing Setup
# -------------------------------------------------------------------
# Specify 3 workers for an N100 CPU (4 cores) environment.
future::plan("multisession", workers = 3)

# -------------------------------------------------------------------
# 3. Define Simulation Function
# -------------------------------------------------------------------
run_one_simulation <- function(sim_id, n, true_ate) {
  # Data Generation (No HTE)
  X1 <- rnorm(n, 0, 1)
  X2 <- rnorm(n, 0, 1)
  X3 <- runif(n, -3, 3)
  X4 <- rbinom(n, 1, 0.5)
  ps_true_logit <- 0.4*X1 - 0.6*(X2^3) + 0.8*X1*X2 - 0.1*(X3^2) - 0.5
  ps_true <- plogis(ps_true_logit)
  T <- rbinom(n, 1, ps_true)
  Y0 <- 5*X1 + 2*(X1^2) - 3*X2 + X2*X3 + 2*X4 + rnorm(n, 0, 2)
  Y1 <- Y0 + true_ate
  Y <- Y1 * T + Y0 * (1 - T)
  sim_data <- data.frame(X1, X2, X3, X4, T, Y)
  X_vars <- c("X1", "X2", "X3", "X4")

  # Pre-create dataframe to store results
  method_names <- c("dml_bart")
  results_df <- data.frame(
    sim_id = rep(sim_id, length(method_names)),
    Method = method_names,
    Estimated_ATE = NA_real_,
    lower_ci = NA_real_,
    upper_ci = NA_real_
  )
  
  # --- Apply Methodologies ---
  
  # IX: DML (BART)
  try({
    dml_data_ml_bart <- DoubleMLData$new(data = sim_data, y_col = "Y", d_cols = "T", x_cols = X_vars)
    ml_l_bart_dml <- lrn("regr.bart")
    ml_m_bart_dml <- lrn("classif.bart")
    dml_model_ml_bart <- DoubleMLPLR$new(data = dml_data_ml_bart, ml_l = ml_l_bart_dml, ml_m = ml_m_bart_dml, n_folds = 5)
    dml_model_ml_bart$fit(store_models = FALSE)
    res_bart_dml <- c(dml_model_ml_bart$coef, dml_model_ml_bart$confint())
    results_df[results_df$Method == "dml_bart", 3:5] <- res_bart_dml
  }, silent = TRUE)

  return(results_df)
}

# -------------------------------------------------------------------
# 4. Run Simulation
# -------------------------------------------------------------------
set.seed(2025)
n_simulations <- 100
n_obs <- 5000
true_ate <- 5

pb <- progress_bar$new(
  format = "  Simulating [:bar] :percent in :elapsed | ETA: :eta",
  total = n_simulations, clear = FALSE, width = 60)

all_results <- list()
for (i in 1:n_simulations) {
  all_results[[i]] <- run_one_simulation(sim_id = i, n = n_obs, true_ate = true_ate)
  pb$tick()
}

final_results_df <- do.call(rbind, all_results) %>%
  unnest(cols = c(Estimated_ATE, lower_ci, upper_ci))

# -------------------------------------------------------------------
# 5. Calculate Performance Metrics and Summarize Results
# -------------------------------------------------------------------
summary_stats <- final_results_df %>%
  group_by(Method) %>%
  summarise(
    Mean_ATE = mean(Estimated_ATE, na.rm = TRUE),
    Bias = mean(Estimated_ATE - true_ate, na.rm = TRUE),
    Std_Dev_ATE = sd(Estimated_ATE, na.rm = TRUE),
    Coverage_Rate = mean(lower_ci <= true_ate & upper_ci >= true_ate, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  ungroup() %>%
  mutate(Method = case_when(
    Method == "dml_bart" ~ "IX. DML (BART)"
  )) %>%
  mutate(Method = fct_relevel(Method,
                              "IX. DML (BART)")) %>%
  arrange(Method)

# --- Console Output ---
cat("\n--- Simulation Performance Summary ---\n")
print(paste("Number of Simulations:", n_simulations))
print(paste("Sample Size (n):", n_obs))
print(paste("True Average Treatment Effect (ATE):", true_ate))
print(as.data.frame(summary_stats))

# --- Save to File ---
sink("dml_bart_summary_results.txt")
cat("--- Simulation Performance Summary ---\n")
print(paste("Number of Simulations:", n_simulations))
print(paste("Sample Size (n):", n_obs))
print(paste("True Average Treatment Effect (ATE):", true_ate))
print(as.data.frame(summary_stats))
sink()
cat("\nSummary results saved to dml_bart_summary_results.txt\n")
