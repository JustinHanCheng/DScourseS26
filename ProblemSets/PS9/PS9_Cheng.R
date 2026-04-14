get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0) {
    return(getwd())
  }
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}

output_dir <- get_script_dir()
lib_dir <- file.path(output_dir, "r_libs")
dir.create(lib_dir, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib_dir, .libPaths()))

ensure_remotes <- function() {
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org", lib = lib_dir)
  }
}

install_version_if_needed <- function(pkg, version) {
  installed <- requireNamespace(pkg, quietly = TRUE)
  installed_version <- if (installed) as.character(packageVersion(pkg)) else NA_character_
  if (!installed || !identical(installed_version, version)) {
    ensure_remotes()
    remotes::install_version(
      pkg,
      version = version,
      repos = "https://cloud.r-project.org",
      lib = lib_dir,
      dependencies = FALSE,
      upgrade = "never"
    )
  }
}

install_version_if_needed("vctrs", "0.6.5")
install_version_if_needed("slider", "0.2.2")
install_version_if_needed("furrr", "0.2.3")
install_version_if_needed("rsample", "1.0.0")

suppressPackageStartupMessages({
  library(recipes)
  library(rsample)
  library(glmnet)
})

SEED <- 123456
SUMMARY_PATH <- file.path(output_dir, "PS9_Cheng_results.txt")

housing <- read.table(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
  header = FALSE
)
names(housing) <- c(
  "crim", "zn", "indus", "chas", "nox", "rm", "age",
  "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"
)

set.seed(SEED)
housing_split <- initial_split(housing, prop = 0.8)
housing_train <- training(housing_split)
housing_test <- testing(housing_split)

housing_recipe <- recipe(medv ~ ., data = housing)
housing_recipe <- step_log(housing_recipe, all_outcomes())
housing_recipe <- step_bin2factor(housing_recipe, chas)
housing_recipe <- step_interact(
  housing_recipe,
  terms = ~ crim:zn:indus:rm:age:rad:tax:ptratio:b:lstat:dis:nox
)
housing_recipe <- step_poly(
  housing_recipe,
  crim, zn, indus, rm, age, rad, tax, ptratio, b, lstat, dis, nox,
  degree = 6
)

housing_prep <- prep(housing_recipe, housing_train, retain = TRUE)
housing_train_prepped <- juice(housing_prep)
housing_test_prepped <- bake(housing_prep, new_data = housing_test)

housing_train_x <- housing_train_prepped[, setdiff(names(housing_train_prepped), "medv"), drop = FALSE]
housing_test_x <- housing_test_prepped[, setdiff(names(housing_test_prepped), "medv"), drop = FALSE]
housing_train_y <- housing_train_prepped$medv
housing_test_y <- housing_test_prepped$medv

x_train_mat <- model.matrix(medv ~ ., data = housing_train_prepped)[, -1]
x_test_mat <- model.matrix(medv ~ ., data = housing_test_prepped)[, -1]

set.seed(SEED)
rec_folds <- vfold_cv(housing_train_prepped, v = 6)
foldid <- integer(nrow(housing_train_prepped))
all_rows <- seq_len(nrow(housing_train_prepped))
for (i in seq_len(nrow(rec_folds))) {
  analysis_rows <- rec_folds$splits[[i]]$in_id
  assessment_rows <- setdiff(all_rows, analysis_rows)
  foldid[assessment_rows] <- i
}
stopifnot(all(foldid %in% seq_len(6)))

lambda_grid <- 10 ^ seq(-10, 0, length.out = 50)

set.seed(SEED)
lasso_cv <- cv.glmnet(
  x_train_mat,
  housing_train_y,
  alpha = 1,
  lambda = lambda_grid,
  foldid = foldid,
  nfolds = 6
)
lasso_lambda <- as.numeric(lasso_cv$lambda.min)
lasso_train_pred <- as.numeric(predict(lasso_cv, newx = x_train_mat, s = "lambda.min"))
lasso_test_pred <- as.numeric(predict(lasso_cv, newx = x_test_mat, s = "lambda.min"))
lasso_train_rmse <- sqrt(mean((housing_train_y - lasso_train_pred) ^ 2))
lasso_test_rmse <- sqrt(mean((housing_test_y - lasso_test_pred) ^ 2))

set.seed(SEED)
ridge_cv <- cv.glmnet(
  x_train_mat,
  housing_train_y,
  alpha = 0,
  lambda = lambda_grid,
  foldid = foldid,
  nfolds = 6
)
ridge_lambda <- as.numeric(ridge_cv$lambda.min)
ridge_train_pred <- as.numeric(predict(ridge_cv, newx = x_train_mat, s = "lambda.min"))
ridge_test_pred <- as.numeric(predict(ridge_cv, newx = x_test_mat, s = "lambda.min"))
ridge_train_rmse <- sqrt(mean((housing_train_y - ridge_train_pred) ^ 2))
ridge_test_rmse <- sqrt(mean((housing_test_y - ridge_test_pred) ^ 2))

summary_lines <- c(
  sprintf("Seed: %d", SEED),
  sprintf("housing_train shape: %d x %d", nrow(housing_train), ncol(housing_train)),
  sprintf("housing_test shape: %d x %d", nrow(housing_test), ncol(housing_test)),
  sprintf("housing_train_prepped shape: %d x %d", nrow(housing_train_prepped), ncol(housing_train_prepped)),
  sprintf("Expanded number of X variables: %d", ncol(housing_train_x)),
  sprintf("Additional X variables: %d", ncol(housing_train_x) - (ncol(housing) - 1)),
  sprintf("LASSO optimal lambda: %.12f", lasso_lambda),
  sprintf("LASSO in-sample RMSE: %.12f", lasso_train_rmse),
  sprintf("LASSO out-of-sample RMSE: %.12f", lasso_test_rmse),
  sprintf("Ridge optimal lambda: %.12f", ridge_lambda),
  sprintf("Ridge in-sample RMSE: %.12f", ridge_train_rmse),
  sprintf("Ridge out-of-sample RMSE: %.12f", ridge_test_rmse)
)

writeLines(summary_lines, SUMMARY_PATH)
writeLines(summary_lines)
