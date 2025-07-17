# -------------------------------------------------------------------
# 0. Package Installation (Run only once)
# -------------------------------------------------------------------
# install.packages(c("tidyverse", "MatchIt", "cobalt", "DoubleML", "mlr3", "mlr3learners", "ranger", "future", "progress", "grf", "bartCause", "mlr3extralearners"))
# install.packages("remotes")
# remotes::install_github("mlr-org/mlr3extralearners@*release")

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
future::plan("multisession", workers = 20)

# -------------------------------------------------------------------
# 3. Define Simulation Function
# -------------------------------------------------------------------
run_one_simulation <- function(sim_id, n, true_ate) {
  # Data Generation (with Unobserved Confounder U - reduced impact)
  X1 <- rnorm(n, 0, 1)
  X2 <- rnorm(n, 0, 1)
  X3 <- runif(n, -3, 3)
  X4 <- rbinom(n, 1, 0.5)
  U <- rnorm(n, 0, 1)  # U is an unobserved confounder
  
  # U affects both T and Y (reduced impact)
  ps_true_logit <- 0.4*X1 - 0.6*(X2^3) + 0.8*X1*X2 - 0.1*(X3^2) + 0.1*U - 0.5 # Reduced U impact
  ps_true <- plogis(ps_true_logit)
  T <- rbinom(n, 1, ps_true)
  
  Y0 <- 5*X1 + 2*(X1^2) - 3*X2 + X2*X3 + 2*X4 + 0.5*U + rnorm(n, 0, 2) # Reduced U impact
  Y1 <- Y0 + true_ate
  Y <- Y1 * T + Y0 * (1 - T)
  
  sim_data <- data.frame(X1, X2, X3, X4, U, T, Y)
  
  # U is not included in the model (assumed unobserved)
  X_vars <- c("X1", "X2", "X3", "X4")

  # Pre-create dataframe to store results
  method_names <- c("reg", "psm", "match_reg", "dr_manual", "dml", "cf", "bart", "xl", "dml_bart")
  results_df <- data.frame(
    sim_id = rep(sim_id, length(method_names)),
    Method = method_names,
    Estimated_ATE = NA_real_,
    lower_ci = NA_real_,
    upper_ci = NA_real_
  )
  
  # --- Apply Methodologies ---
  
  # I: Outcome Regression
  try({
    outcome_model <- lm(Y ~ T + X1 + X2 + X3 + X4, data = sim_data)
    res <- c(coef(outcome_model)["T"], confint(outcome_model, "T"))
    results_df[results_df$Method == "reg", 3:5] <- res
  }, silent = TRUE)

  # II: PSM
  try({
    matched_data_obj <- matchit(T ~ X1 + X2 + X3 + X4, data = sim_data, method = "nearest", distance = "glm", verbose = FALSE)
    matched_sample <- match.data(matched_data_obj)
    psm_model <- lm(Y ~ T, data = matched_sample)
    res <- c(coef(psm_model)["T"], confint(psm_model, "T"))
    results_df[results_df$Method == "psm", 3:5] <- res
  }, silent = TRUE)

  # III: Matching with Regression
  try({
    adj_model <- lm(Y ~ T + X1 + X2 + X3 + X4, data = matched_sample)
    res <- c(coef(adj_model)["T"], confint(adj_model, "T"))
    results_df[results_df$Method == "match_reg", 3:5] <- res
  }, silent = TRUE)

  # IV: Doubly Robust Estimation (Manual)
  try({
    ps_model <- glm(T ~ X1 + X2 + X3 + X4, data = sim_data, family = "binomial")
    ps_manual <- predict(ps_model, type = "response")
    epsilon <- 0.001
    ps_manual_clipped <- pmax(epsilon, pmin(1 - epsilon, ps_manual))
    
    outcome_model_manual <- lm(Y ~ T + X1 + X2 + X3 + X4, data = sim_data)
    mu1 <- predict(outcome_model_manual, newdata = mutate(sim_data, T = 1))
    mu0 <- predict(outcome_model_manual, newdata = mutate(sim_data, T = 0))
    
    y1_hat <- (sim_data$T * sim_data$Y / ps_manual_clipped) - ((sim_data$T - ps_manual_clipped) / ps_manual_clipped) * mu1
    y0_hat <- ((1 - sim_data$T) * sim_data$Y / (1 - ps_manual_clipped)) + ((sim_data$T - ps_manual_clipped) / (1 - ps_manual_clipped)) * mu0
    ate_manual_dr <- mean(y1_hat) - mean(y0_hat)
    if_dr <- y1_hat - y0_hat
    se_manual_dr <- sqrt(var(if_dr, na.rm = TRUE) / n)
    ci_manual_dr <- c(ate_manual_dr - 1.96 * se_manual_dr, ate_manual_dr + 1.96 * se_manual_dr)
    results_df[results_df$Method == "dr_manual", 3:5] <- c(ate_manual_dr, ci_manual_dr)
  }, silent = TRUE)

  # V: DML (Ranger)
  try({
    dml_data_ml <- DoubleMLData$new(data = sim_data, y_col = "Y", d_cols = "T", x_cols = X_vars)
    ml_l_ranger <- lrn("regr.ranger", num.trees = 100)
    ml_m_ranger <- lrn("classif.ranger", num.trees = 100)
    dml_model_ml <- DoubleMLPLR$new(data = dml_data_ml, ml_l = ml_l_ranger, ml_m = ml_m_ranger, n_folds = 5)
    dml_model_ml$fit(store_models = FALSE)
    res <- c(dml_model_ml$coef, dml_model_ml$confint())
    results_df[results_df$Method == "dml", 3:5] <- res
  }, silent = TRUE)
  
  # VI: Causal Forest
  try({
    X_matrix <- as.matrix(sim_data[, X_vars])
    cf_model <- causal_forest(X = X_matrix, Y = sim_data$Y, W = sim_data$T)
    ate_cf_obj <- average_treatment_effect(cf_model, target.sample = "all")
    ate_cf <- ate_cf_obj["estimate"]
    ci_cf <- c(ate_cf - 1.96 * ate_cf_obj["std.err"], ate_cf + 1.96 * ate_cf_obj["std.err"])
    results_df[results_df$Method == "cf", 3:5] <- c(ate_cf, ci_cf)
  }, silent = TRUE)

  # VII: BART
  try({
    bart_model <- bartc(response = Y, treatment = T, confounders = X1 + X2 + X3 + X4, data = sim_data, seed = 2025, n.threads = 1)
    bart_summary <- summary(bart_model)
    ate_bart <- bart_summary$estimates["ate", "estimate"]
    ci_bart <- bart_summary$estimates["ate", c("ci.lower", "ci.upper")]
    results_df[results_df$Method == "bart", 3:5] <- c(ate_bart, ci_bart)
  }, silent = TRUE)

  # VIII: X-learner
  try({
    control_data <- filter(sim_data, T == 0)
    treated_data <- filter(sim_data, T == 1)
    X_control <- control_data[, X_vars]
    X_treated <- treated_data[, X_vars]
    model_y0 <- ranger(Y ~ ., data = control_data[, c("Y", X_vars)], num.trees = 100)
    model_y1 <- ranger(Y ~ ., data = treated_data[, c("Y", X_vars)], num.trees = 100)
    d1 <- treated_data$Y - predict(model_y0, data = X_treated)$predictions
    d0 <- predict(model_y1, data = X_control)$predictions - control_data$Y
    model_tau1 <- ranger(d1 ~ ., data = data.frame(d1 = d1, X_treated), num.trees = 100)
    model_tau0 <- ranger(d0 ~ ., data = data.frame(d0 = d0, X_control), num.trees = 100)
    tau1_hat <- predict(model_tau1, data = sim_data[, X_vars])$predictions
    tau0_hat <- predict(model_tau0, data = sim_data[, X_vars])$predictions
    ps_model_xl <- ranger(T ~ ., data = sim_data[, c("T", X_vars)], probability = TRUE, num.trees = 100)
    g_hat <- predict(ps_model_xl, data = sim_data[, X_vars])$predictions[, 2]
    tau_hat <- g_hat * tau0_hat + (1 - g_hat) * tau1_hat
    ate_xl <- mean(tau_hat)
    se_xl <- sqrt(var(tau_hat, na.rm = TRUE) / n)
    ci_xl <- c(ate_xl - 1.96 * se_xl, ate_xl + 1.96 * se_xl)
    results_df[results_df$Method == "xl", 3:5] <- c(ate_xl, ci_xl)
  }, silent = TRUE)

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
    Method == "reg" ~ "I. Regression",
    Method == "psm" ~ "II. PSM",
    Method == "match_reg" ~ "III. Matching + Regression",
    Method == "dr_manual" ~ "IV. Double Robust (Manual)",
    Method == "dml" ~ "V. DML (Ranger)",
    Method == "cf" ~ "VI. Causal Forest",
    Method == "bart" ~ "VII. BART",
    Method == "xl" ~ "VIII. X-learner",
    Method == "dml_bart" ~ "IX. DML (BART)"
  )) %>%
  mutate(Method = fct_relevel(Method,
                              "I. Regression", "II. PSM", "III. Matching + Regression",
                              "IV. Double Robust (Manual)", "V. DML (Ranger)",
                              "VI. Causal Forest", "VII. BART", "VIII. X-learner", "IX. DML (BART)")) %>%
  arrange(Method)

# --- Console Output ---
cat("\n--- Simulation Performance Summary (with Unknown Confounder - Reduced Impact) ---\n")
print(paste("Number of Simulations:", n_simulations))
print(paste("Sample Size (n):", n_obs))
print(paste("True Average Treatment Effect (ATE):", true_ate))
print(as.data.frame(summary_stats))
sink("simulation_summary_results_uc_modified.txt")
cat("--- Simulation Performance Summary (with Unknown Confounder - Reduced Impact) ---\n")
print(paste("Number of Simulations:", n_simulations))
print(paste("Sample Size (n):", n_obs))
print(paste("True Average Treatment Effect (ATE):", true_ate))
print(as.data.frame(summary_stats))
sink()
cat("\nSummary results saved to simulation_summary_results_uc_modified.txt\n")

# -------------------------------------------------------------------
# 6. Visualize Results (Boxplot)
# -------------------------------------------------------------------
p <- ggplot(final_results_df, aes(x = fct_reorder(Method, Estimated_ATE, .fun=median, .desc=FALSE, .na_rm = TRUE), y = Estimated_ATE, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = true_ate, linetype = "dashed", color = "red", linewidth = 1.2) +
  annotate("text", x = 0.8, y = true_ate + 0.1, label = paste("True ATE =", true_ate), color = "red", size = 5, hjust=0) +
  labs(title = "Distribution of ATE Estimates with Unknown Confounder (Reduced Impact)",
       x = "Methodology",
       y = "Estimated Average Treatment Effect (ATE)") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 20))

ggsave("simulation_results_boxplot_uc_modified.png", plot = p, width = 12, height = 8, dpi = 300)

cat("\nBoxplot graph saved to simulation_results_boxplot_uc_modified.png\n")
