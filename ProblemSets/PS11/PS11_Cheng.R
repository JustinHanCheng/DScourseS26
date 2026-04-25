# =============================================================================
# PS11_Cheng.R
# Reproducibility driver for the rough-draft final project.
#
# This script reproduces the numerical entries that PS11_Cheng.tex reports in
# Table 2 by re-running the tuned classifiers from Problem Set 10 on the UCI
# Adult Income data. It is intentionally a thin wrapper around the PS10
# pipeline so that the final-project draft can be regenerated end to end.
#
# Run from this folder:
#   Rscript PS11_Cheng.R
#
# It writes:
#   PS11_Cheng_results.txt     numeric summary used by PS11_Cheng.tex
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(forcats)
  library(doParallel)
})

set.seed(100)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

cn <- c("age","workclass","fnlwgt","education","education.num",
        "marital.status","occupation","relationship","race","sex",
        "capital.gain","capital.loss","hours","native.country","high.earner")

income <- readr::read_csv(url, col_names = cn, na = "?", show_col_types = FALSE) |>
  drop_na() |>
  mutate(high.earner = factor(if_else(trimws(high.earner) == ">50K", 1L, 0L))) |>
  mutate(across(c(workclass, education, marital.status, occupation,
                  relationship, race, sex), as.factor)) |>
  select(-native.country, -fnlwgt, -education.num)

# Same level-collapse as in PS10
income <- income |>
  mutate(
    marital.status = fct_collapse(marital.status,
      "Married"      = c("Married-AF-spouse","Married-civ-spouse","Married-spouse-absent"),
      "Never-Married"= c("Never-married"),
      "Other"        = c("Divorced","Separated","Widowed")),
    race = fct_collapse(race,
      "White" = c("White"),
      "Black" = c("Black"),
      "Other" = c("Amer-Indian-Eskimo","Asian-Pac-Islander","Other"))
  )

income_split <- initial_split(income, prop = 0.80)
income_train <- training(income_split)
income_test  <- testing(income_split)

# -----------------------------------------------------------------------------
# Shared recipe
# -----------------------------------------------------------------------------
rec <- recipe(high.earner ~ ., data = income_train) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

folds <- vfold_cv(income_train, v = 3)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
cl <- makePSOCKcluster(max(1L, parallel::detectCores() - 1L))
registerDoParallel(cl)
on.exit(stopCluster(cl), add = TRUE)

logit <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet") |> set_mode("classification")

tree <- decision_tree(cost_complexity = tune(), tree_depth = tune(),
                      min_n = tune()) |>
  set_engine("rpart") |> set_mode("classification")

nnet_spec <- mlp(hidden_units = tune(), penalty = tune()) |>
  set_engine("nnet") |> set_mode("classification")

knn <- nearest_neighbor(neighbors = tune()) |>
  set_engine("kknn") |> set_mode("classification")

svm_rbf_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) |>
  set_engine("kernlab") |> set_mode("classification")

# Grids: real-valued -> grid_regular(); integer-valued -> tibble()
g_logit <- grid_regular(penalty(), levels = 50)

g_tree <- crossing(
  grid_regular(cost_complexity(range = c(0.001, 0.2), trans = NULL), levels = 5),
  tibble(min_n      = c(10, 20, 30, 40, 50)),
  tibble(tree_depth = c(5, 10, 15, 20))
)

g_nn <- crossing(
  tibble(hidden_units = 1:10),
  grid_regular(penalty(), levels = 10)
)

g_knn <- tibble(neighbors = 1:30)

g_svm <- crossing(
  tibble(cost      = c(2^-2, 2^-1, 1, 2, 4, 1024)),
  tibble(rbf_sigma = c(2^-2, 2^-1, 1, 2, 4, 1024))
)

run <- function(spec, grid) {
  workflow() |>
    add_recipe(rec) |>
    add_model(spec) |>
    tune_grid(resamples = folds, grid = grid,
              metrics = metric_set(accuracy)) |>
    (\(t) {
      best <- select_best(t, metric = "accuracy")
      cv   <- collect_metrics(t) |>
              filter(.config == best$.config) |>
              pull(mean)
      fit  <- finalize_workflow(workflow() |> add_recipe(rec) |> add_model(spec),
                                best) |>
              last_fit(income_split)
      list(best = best,
           cv   = cv,
           test = collect_metrics(fit) |>
                  filter(.metric == "accuracy") |> pull(.estimate))
    })()
}

res <- list(
  logit = run(logit,        g_logit),
  tree  = run(tree,         g_tree),
  nnet  = run(nnet_spec,    g_nn),
  knn   = run(knn,          g_knn),
  svm   = run(svm_rbf_spec, g_svm)
)

# -----------------------------------------------------------------------------
# Write summary
# -----------------------------------------------------------------------------
sink("PS11_Cheng_results.txt")
cat("PS11 results -- Adult Income classification\n")
cat("seed = 100; 80/20 split; 3-fold CV\n\n")
for (nm in names(res)) {
  cat("---", nm, "---\n")
  print(res[[nm]]$best)
  cat(sprintf("CV accuracy:   %.4f\n",   res[[nm]]$cv))
  cat(sprintf("Test accuracy: %.4f\n\n", res[[nm]]$test))
}
sink()

cat("Wrote PS11_Cheng_results.txt\n")
