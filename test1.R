# -------------------------------------------------------------------
# 0. 패키지 설치 (최초 1회만 실행)
# -------------------------------------------------------------------
# install.packages(c("tidyverse", "MatchIt", "cobalt", "DoubleML", "mlr3", "mlr3learners", "ranger", "grf", "bartCause"))

# -------------------------------------------------------------------
# 1. 패키지 로드
# -------------------------------------------------------------------
library(tidyverse)
library(MatchIt)
library(cobalt)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(ranger)
library(future)
library(grf)
library(bartCause)

future::plan("multisession")

# -------------------------------------------------------------------
# 2. 시뮬레이션 설정
# -------------------------------------------------------------------
set.seed(2025)
n <- 5000
true_ate <- 5

# -------------------------------------------------------------------
# 3. 데이터 생성 함수 (Model Misspecification 포함)
# -------------------------------------------------------------------
generate_data <- function(n, true_ate) {
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

  data.frame(X1, X2, X3, X4, T, Y)
}

sim_data <- generate_data(n, true_ate)
X_vars <- c("X1", "X2", "X3", "X4")

# -------------------------------------------------------------------
# 4. 인과추론 방법론 적용 (신뢰구간 계산 포함)
# -------------------------------------------------------------------

## 방법 I: 결과 회귀분석
outcome_model <- lm(Y ~ T + X1 + X2 + X3 + X4, data = sim_data)
ate_regression <- coef(outcome_model)["T"]
ci_regression <- confint(outcome_model, "T", level = 0.95)

## 방법 II: 성향점수 매칭 (PSM)
matched_data_obj <- matchit(T ~ X1 + X2 + X3 + X4, data = sim_data, method = "nearest", distance = "glm")
matched_sample <- match.data(matched_data_obj)
psm_model <- lm(Y ~ T, data = matched_sample)
ate_psm <- coef(psm_model)["T"]
ci_psm <- confint(psm_model, "T", level = 0.95)

## 방법 III: 매칭 후 회귀분석
adj_model <- lm(Y ~ T + X1 + X2 + X3 + X4, data = matched_sample)
ate_match_reg <- coef(adj_model)["T"]
ci_match_reg <- confint(adj_model, "T", level = 0.95)

## 방법 IV: 이중 강건 추정법 (직접 계산)
ps_model <- glm(T ~ X1 + X2 + X3 + X4, data = sim_data, family = "binomial")
ps_manual <- predict(ps_model, type = "response")
outcome_model_manual <- lm(Y ~ T + X1 + X2 + X3 + X4, data = sim_data)
mu1 <- predict(outcome_model_manual, newdata = mutate(sim_data, T = 1))
mu0 <- predict(outcome_model_manual, newdata = mutate(sim_data, T = 0))
y1_hat <- (sim_data$T * sim_data$Y / ps_manual) - ((sim_data$T - ps_manual) / ps_manual) * mu1
y0_hat <- ((1 - sim_data$T) * sim_data$Y / (1 - ps_manual)) + ((sim_data$T - ps_manual) / (1 - ps_manual)) * mu0
ate_manual_dr <- mean(y1_hat) - mean(y0_hat)
if_dr <- y1_hat - y0_hat
se_manual_dr <- sqrt(var(if_dr) / n)
ci_manual_dr <- c(ate_manual_dr - 1.96 * se_manual_dr, ate_manual_dr + 1.96 * se_manual_dr)

## 방법 V (고급): 이중/탈편향 머신러닝 (DML)
dml_data_ml <- DoubleMLData$new(data = sim_data, y_col = "Y", d_cols = "T", x_cols = X_vars)
ml_g_ranger <- lrn("regr.ranger", num.trees = 100)
ml_m_ranger <- lrn("classif.ranger", num.trees = 100)
dml_model_ml <- DoubleMLPLR$new(data = dml_data_ml, ml_l = ml_g_ranger, ml_m = ml_m_ranger, n_folds = 5)
dml_model_ml$fit()
ate_dml <- dml_model_ml$coef
ci_dml <- dml_model_ml$confint()

## 방법 VI (고급): Causal Forest (grf)
X_matrix <- as.matrix(sim_data[, X_vars])
Y_vec <- sim_data$Y
T_vec <- sim_data$T
cf_model <- causal_forest(X = X_matrix, Y = Y_vec, W = T_vec)
ate_cf_obj <- average_treatment_effect(cf_model, target.sample = "all")
ate_cf <- ate_cf_obj["estimate"]
ci_cf <- c(ate_cf_obj["estimate"] - 1.96 * ate_cf_obj["std.err"],
           ate_cf_obj["estimate"] + 1.96 * ate_cf_obj["std.err"])

## 방법 VII (고급): Bayesian Additive Regression Trees (BART)
bart_model <- bartc(
  response = Y,
  treatment = T,
  confounders = X1 + X2 + X3 + X4,
  data = sim_data,
  seed = 2025
)
bart_summary <- summary(bart_model)
ate_bart <- bart_summary$estimates["ate", "estimate"]
ci_bart <- bart_summary$estimates["ate", c("ci.lower", "ci.upper")]

## 방법 VIII (고급): X-learner (Manual Implementation with Ranger)
control_data <- filter(sim_data, T == 0)
treated_data <- filter(sim_data, T == 1)
X_control <- control_data[, X_vars]
X_treated <- treated_data[, X_vars]
model_y0 <- ranger(Y ~ ., data = control_data[, c("Y", X_vars)], num.trees = 100, seed = 2025)
model_y1 <- ranger(Y ~ ., data = treated_data[, c("Y", X_vars)], num.trees = 100, seed = 2025)
d1 <- treated_data$Y - predict(model_y0, data = X_treated)$predictions
d0 <- predict(model_y1, data = X_control)$predictions - control_data$Y
model_tau1 <- ranger(d1 ~ ., data = data.frame(d1 = d1, X_treated), num.trees = 100, seed = 2025)
model_tau0 <- ranger(d0 ~ ., data = data.frame(d0 = d0, X_control), num.trees = 100, seed = 2025)
tau1_hat <- predict(model_tau1, data = sim_data[, X_vars])$predictions
tau0_hat <- predict(model_tau0, data = sim_data[, X_vars])$predictions
ps_model_xl <- ranger(T ~ ., data = sim_data[, c("T", X_vars)], probability = TRUE, num.trees = 100, seed = 2025)
g_hat <- predict(ps_model_xl, data = sim_data[, X_vars])$predictions[, 2]
tau_hat <- g_hat * tau0_hat + (1 - g_hat) * tau1_hat
ate_xl <- mean(tau_hat)
se_xl <- sqrt(var(tau_hat) / n)
ci_xl <- c(ate_xl - 1.96 * se_xl, ate_xl + 1.96 * se_xl)

# -------------------------------------------------------------------
# 5. 최종 결과 종합 및 시각화
# -------------------------------------------------------------------
results_df <- tibble(
  Method = c("I. Outcome Regression",
             "II. Propensity Score Matching",
             "III. Matching w/ Regression",
             "IV. Doubly Robust (Manual)",
             "V. Double/Debiased ML (DML)",
             "VI. Causal Forest (grf)",
             "VII. BART (bartCause)",
             "VIII. X-learner (Ranger)"),
  Estimated_ATE = c(ate_regression,
                    ate_psm,
                    ate_match_reg,
                    ate_manual_dr,
                    ate_dml,
                    ate_cf,
                    ate_bart,
                    ate_xl),
  lower_ci = c(ci_regression[1],
               ci_psm[1],
               ci_match_reg[1],
               ci_manual_dr[1],
               ci_dml[1],
               ci_cf[1],
               ci_bart[1],
               ci_xl[1]),
  upper_ci = c(ci_regression[2],
               ci_psm[2],
               ci_match_reg[2],
               ci_manual_dr[2],
               ci_dml[2],
               ci_cf[2],
               ci_bart[2],
               ci_xl[2])
) %>%
  unnest(cols = c(lower_ci, upper_ci))

results_df$Bias <- results_df$Estimated_ATE - true_ate

cat("\n--- Final Results Summary (with 95% CI) ---\n")
print(paste("True Average Treatment Effect (ATE):", true_ate))
results_df %>%
  mutate(across(where(is.numeric), ~round(., 3))) %>%
  print(n = 8)

# 신뢰구간을 포함한 시각화
ggplot(results_df, aes(x = Method, y = Estimated_ATE, fill = Method)) +
  geom_bar(stat = "identity", width = 0.7, color = "black", alpha = 0.7) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.2, linewidth = 1, color = "gray20") +
  geom_hline(yintercept = true_ate, linetype = "dashed", color = "red", linewidth = 1.2) +
  geom_text(aes(label = round(Estimated_ATE, 2)), vjust = -0.5, size = 4, color = "black") +
  annotate("text", x = 0.8, y = true_ate + 0.1, label = paste("True ATE =", true_ate), color = "red", size = 5, hjust=0) +
  labs(title = "Causal Inference Method Performance with 95% Confidence Intervals",
       x = "Methodology",
       y = "Estimated Average Treatment Effect (ATE)") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 30, hjust = 1))