# Reproduces Econ 5253 Problem Set 9 for Cheng.

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)

  if (length(file_arg) == 0) {
    return(getwd())
  }

  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}

output_dir <- get_script_dir()
summary_path <- file.path(output_dir, "PS9_Cheng_results.txt")

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidymodels)
  library(glmnet)
})

SEED <- 123456

set.seed(SEED)

housing <- read_table(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
  col_names = FALSE,
  show_col_types = FALSE
)

names(housing) <- c(
  "crim", "zn", "indus", "chas", "nox", "rm", "age",
  "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"
)

housing_split <- initial_split(housing, prop = 0.8)
housing_train <- training(housing_split)
housing_test <- testing(housing_split)

housing_recipe <- recipe(medv ~ ., data = housing) %>%
  step_log(all_outcomes()) %>%
  step_bin2factor(chas) %>%
  step_interact(
    terms = ~ crim:zn:indus:rm:age:rad:tax:ptratio:b:lstat:dis:nox
  ) %>%
  step_poly(
    crim, zn, indus, rm, age, rad, tax, ptratio, b, lstat, dis, nox,
    degree = 6
  )

housing_prep <- housing_recipe %>%
  prep(training = housing_train, retain = TRUE)

housing_train_prepped <- juice(housing_prep)
housing_test_prepped <- bake(housing_prep, new_data = housing_test)

housing_train_x <- select(housing_train_prepped, -medv)
housing_test_x <- select(housing_test_prepped, -medv)
housing_train_y <- select(housing_train_prepped, medv)
housing_test_y <- select(housing_test_prepped, medv)

lambda_grid <- grid_regular(penalty(), levels = 50)

set.seed(SEED)
housing_folds <- vfold_cv(housing_train_prepped, v = 6)

lasso_spec <- linear_reg(
  penalty = tune(),
  mixture = 1
) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

lasso_workflow <- workflow() %>%
  add_formula(medv ~ .) %>%
  add_model(lasso_spec)

lasso_res <- tune_grid(
  lasso_workflow,
  resamples = housing_folds,
  grid = lambda_grid,
  metrics = metric_set(rmse)
)

lasso_best <- select_best(lasso_res, metric = "rmse")

final_lasso <- finalize_workflow(lasso_workflow, lasso_best)

lasso_fit <- fit(final_lasso, data = housing_train_prepped)

lasso_train_rmse <- lasso_fit %>%
  predict(housing_train_prepped) %>%
  mutate(truth = housing_train_prepped$medv) %>%
  rmse(truth = truth, estimate = .pred) %>%
  pull(.estimate)

lasso_test_rmse <- lasso_fit %>%
  predict(housing_test_prepped) %>%
  mutate(truth = housing_test_prepped$medv) %>%
  rmse(truth = truth, estimate = .pred) %>%
  pull(.estimate)

ridge_spec <- linear_reg(
  penalty = tune(),
  mixture = 0
) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

ridge_workflow <- workflow() %>%
  add_formula(medv ~ .) %>%
  add_model(ridge_spec)

ridge_res <- tune_grid(
  ridge_workflow,
  resamples = housing_folds,
  grid = lambda_grid,
  metrics = metric_set(rmse)
)

ridge_best <- select_best(ridge_res, metric = "rmse")

final_ridge <- finalize_workflow(ridge_workflow, ridge_best)

ridge_fit <- fit(final_ridge, data = housing_train_prepped)

ridge_train_rmse <- ridge_fit %>%
  predict(housing_train_prepped) %>%
  mutate(truth = housing_train_prepped$medv) %>%
  rmse(truth = truth, estimate = .pred) %>%
  pull(.estimate)

ridge_test_rmse <- ridge_fit %>%
  predict(housing_test_prepped) %>%
  mutate(truth = housing_test_prepped$medv) %>%
  rmse(truth = truth, estimate = .pred) %>%
  pull(.estimate)

summary_lines <- c(
  sprintf("Seed: %d", SEED),
  sprintf("housing_train shape: %d x %d", nrow(housing_train), ncol(housing_train)),
  sprintf("housing_test shape: %d x %d", nrow(housing_test), ncol(housing_test)),
  sprintf(
    "housing_train_prepped shape: %d x %d",
    nrow(housing_train_prepped),
    ncol(housing_train_prepped)
  ),
  sprintf("Expanded number of X variables: %d", ncol(housing_train_x)),
  sprintf("Additional X variables: %d", ncol(housing_train_x) - (ncol(housing) - 1)),
  sprintf("LASSO optimal lambda: %.12f", lasso_best$penalty[[1]]),
  sprintf("LASSO in-sample RMSE: %.12f", lasso_train_rmse),
  sprintf("LASSO out-of-sample RMSE: %.12f", lasso_test_rmse),
  sprintf("Ridge optimal lambda: %.12f", ridge_best$penalty[[1]]),
  sprintf("Ridge in-sample RMSE: %.12f", ridge_train_rmse),
  sprintf("Ridge out-of-sample RMSE: %.12f", ridge_test_rmse)
)

writeLines(summary_lines, summary_path)
writeLines(summary_lines)
