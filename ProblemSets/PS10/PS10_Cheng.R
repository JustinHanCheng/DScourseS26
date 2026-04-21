# Reproduces Econ 5253 Problem Set 10 for Cheng.

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)

  if (length(file_arg) == 0) {
    return(getwd())
  }

  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}

output_dir <- get_script_dir()
summary_path <- file.path(output_dir, "PS10_Cheng_results.txt")
rds_path <- file.path(output_dir, "PS10_Cheng_results.rds")

ensure_pkg <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
}

ensure_pkg(c(
  "tidyverse", "tidymodels", "magrittr", "modelsummary",
  "rpart", "e1071", "kknn", "nnet", "kernlab", "glmnet",
  "doParallel"
))

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(magrittr)
  library(modelsummary)
  library(rpart)
  library(e1071)
  library(kknn)
  library(nnet)
  library(kernlab)
  library(glmnet)
  library(doParallel)
})

set.seed(100)

income <- read_csv(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
  col_names = FALSE,
  show_col_types = FALSE
)
names(income) <- c(
  "age", "workclass", "fnlwgt", "education", "education.num",
  "marital.status", "occupation", "relationship", "race", "sex",
  "capital.gain", "capital.loss", "hours", "native.country", "high.earner"
)

income %<>% select(-native.country, -fnlwgt, -education.num)
income %<>% mutate(across(
  c(age, hours, capital.gain, capital.loss), as.numeric
))
income %<>% mutate(across(
  c(high.earner, education, marital.status, race, workclass,
    occupation, relationship, sex),
  as.factor
))

income %<>% mutate(
  education = fct_collapse(
    education,
    Advanced    = c("Masters", "Doctorate", "Prof-school"),
    Bachelors   = c("Bachelors"),
    SomeCollege = c("Some-college", "Assoc-acdm", "Assoc-voc"),
    HSgrad      = c("HS-grad", "12th"),
    HSdrop      = c("11th", "9th", "7th-8th", "1st-4th", "10th", "5th-6th", "Preschool")
  ),
  marital.status = fct_collapse(
    marital.status,
    Married      = c("Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"),
    Divorced     = c("Divorced", "Separated"),
    Widowed      = c("Widowed"),
    NeverMarried = c("Never-married")
  ),
  race = fct_collapse(
    race,
    White = c("White"),
    Black = c("Black"),
    Asian = c("Asian-Pac-Islander"),
    Other = c("Other", "Amer-Indian-Eskimo")
  ),
  workclass = fct_collapse(
    workclass,
    Private = c("Private"),
    SelfEmp = c("Self-emp-not-inc", "Self-emp-inc"),
    Gov     = c("Federal-gov", "Local-gov", "State-gov"),
    Other   = c("Without-pay", "Never-worked", "?")
  ),
  occupation = fct_collapse(
    occupation,
    BlueCollar  = c("?", "Craft-repair", "Farming-fishing", "Handlers-cleaners",
                    "Machine-op-inspct", "Transport-moving"),
    WhiteCollar = c("Adm-clerical", "Exec-managerial", "Prof-specialty",
                    "Sales", "Tech-support"),
    Services    = c("Armed-Forces", "Other-service", "Priv-house-serv",
                    "Protective-serv")
  )
)

income_split <- initial_split(income, prop = 0.8)
income_train <- training(income_split)
income_test  <- testing(income_split)

rec_folds <- vfold_cv(income_train, v = 3)

f <- high.earner ~ education + marital.status + race + workclass + occupation +
  relationship + sex + age + capital.gain + capital.loss + hours

acc_metric <- metric_set(accuracy)

n_cores <- max(1, parallel::detectCores() - 1)
cl <- makePSOCKcluster(n_cores)
registerDoParallel(cl)
on.exit(stopCluster(cl), add = TRUE)

#####################
# Logistic regression
#####################
message("Starting LOGIT")

tune_logit_spec <- logistic_reg(
  penalty = tune(),
  mixture = 1
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

logit_grid <- grid_regular(penalty(), levels = 50)

logit_wf <- workflow() %>%
  add_model(tune_logit_spec) %>%
  add_formula(f)

logit_res <- logit_wf %>%
  tune_grid(
    resamples = rec_folds,
    grid = logit_grid,
    metrics = acc_metric
  )

best_logit <- select_best(logit_res, metric = "accuracy")
final_logit <- finalize_workflow(logit_wf, best_logit)
logit_test <- last_fit(final_logit, income_split) %>% collect_metrics()

logit_ans <- show_best(logit_res, metric = "accuracy") %>%
  slice(1) %>%
  left_join(logit_test %>% slice(1), by = c(".metric", ".estimator")) %>%
  mutate(alg = "logit") %>%
  select(-starts_with(".config"))

#####################
# Decision tree
#####################
message("Starting TREE")

tune_tree_spec <- decision_tree(
  min_n = tune(),
  tree_depth = tune(),
  cost_complexity = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

tree_cc <- grid_regular(
  cost_complexity(range = c(0.001, 0.2), trans = NULL),
  levels = 5
)
tree_grid <- tidyr::crossing(
  tree_cc,
  min_n = seq(10, 50, by = 10),
  tree_depth = seq(5, 20, by = 5)
)

tree_wf <- workflow() %>%
  add_model(tune_tree_spec) %>%
  add_formula(f)

tree_res <- tree_wf %>%
  tune_grid(
    resamples = rec_folds,
    grid = tree_grid,
    metrics = acc_metric
  )

best_tree <- select_best(tree_res, metric = "accuracy")
final_tree <- finalize_workflow(tree_wf, best_tree)
tree_test <- last_fit(final_tree, income_split) %>% collect_metrics()

tree_ans <- show_best(tree_res, metric = "accuracy") %>%
  slice(1) %>%
  left_join(tree_test %>% slice(1), by = c(".metric", ".estimator")) %>%
  mutate(alg = "tree") %>%
  select(-starts_with(".config"))

#####################
# Neural network
#####################
message("Starting NNET")

tune_nnet_spec <- mlp(
  hidden_units = tune(),
  penalty = tune()
) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nnet_grid <- tidyr::crossing(
  hidden_units = 1:10,
  grid_regular(penalty(), levels = 10)
)

nnet_wf <- workflow() %>%
  add_model(tune_nnet_spec) %>%
  add_formula(f)

nnet_res <- nnet_wf %>%
  tune_grid(
    resamples = rec_folds,
    grid = nnet_grid,
    metrics = acc_metric
  )

best_nnet <- select_best(nnet_res, metric = "accuracy")
final_nnet <- finalize_workflow(nnet_wf, best_nnet)
nnet_test <- last_fit(final_nnet, income_split) %>% collect_metrics()

nnet_ans <- show_best(nnet_res, metric = "accuracy") %>%
  slice(1) %>%
  left_join(nnet_test %>% slice(1), by = c(".metric", ".estimator")) %>%
  mutate(alg = "nnet") %>%
  select(-starts_with(".config"))

#####################
# k-Nearest Neighbors
#####################
message("Starting KNN")

tune_knn_spec <- nearest_neighbor(
  neighbors = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

knn_grid <- tibble(neighbors = seq(1, 30))

knn_wf <- workflow() %>%
  add_model(tune_knn_spec) %>%
  add_formula(f)

knn_res <- knn_wf %>%
  tune_grid(
    resamples = rec_folds,
    grid = knn_grid,
    metrics = acc_metric
  )

best_knn <- select_best(knn_res, metric = "accuracy")
final_knn <- finalize_workflow(knn_wf, best_knn)
knn_test <- last_fit(final_knn, income_split) %>% collect_metrics()

knn_ans <- show_best(knn_res, metric = "accuracy") %>%
  slice(1) %>%
  left_join(knn_test %>% slice(1), by = c(".metric", ".estimator")) %>%
  mutate(alg = "knn") %>%
  select(-starts_with(".config"))

#####################
# SVM (RBF kernel)
#####################
message("Starting SVM")

tune_svm_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_grid <- tidyr::crossing(
  cost      = c(2^(-2), 2^(-1), 2^0, 2^1, 2^2, 2^10),
  rbf_sigma = c(2^(-2), 2^(-1), 2^0, 2^1, 2^2, 2^10)
)

svm_wf <- workflow() %>%
  add_model(tune_svm_spec) %>%
  add_formula(f)

svm_res <- svm_wf %>%
  tune_grid(
    resamples = rec_folds,
    grid = svm_grid,
    metrics = acc_metric
  )

best_svm <- select_best(svm_res, metric = "accuracy")
final_svm <- finalize_workflow(svm_wf, best_svm)
svm_test <- last_fit(final_svm, income_split) %>% collect_metrics()

svm_ans <- show_best(svm_res, metric = "accuracy") %>%
  slice(1) %>%
  left_join(svm_test %>% slice(1), by = c(".metric", ".estimator")) %>%
  mutate(alg = "svm") %>%
  select(-starts_with(".config"))

#####################
# Combine
#####################
all_ans <- bind_rows(logit_ans, tree_ans, nnet_ans, knn_ans, svm_ans)
saveRDS(all_ans, rds_path)

fmt_num <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x)) return("NA")
  formatC(x, format = "f", digits = 12)
}

pick <- function(df, col) {
  if (col %in% names(df)) df[[col]][[1]] else NA_real_
}

alg_label <- c(
  logit = "Logistic regression (glmnet)",
  tree  = "Decision tree (rpart)",
  nnet  = "Neural network (nnet)",
  knn   = "k-Nearest Neighbors (kknn)",
  svm   = "SVM RBF (kernlab)"
)

summary_lines <- c(
  "Problem Set 10 results (seed = 100, 80/20 split, 3-fold CV, metric = accuracy)",
  sprintf("Training observations: %d", nrow(income_train)),
  sprintf("Test observations: %d", nrow(income_test)),
  ""
)

for (row_idx in seq_len(nrow(all_ans))) {
  row <- all_ans[row_idx, ]
  summary_lines <- c(summary_lines, paste0("== ", alg_label[[row$alg]], " =="))
  if (row$alg == "logit") {
    summary_lines <- c(summary_lines, sprintf("penalty: %s", fmt_num(pick(row, "penalty"))))
  } else if (row$alg == "tree") {
    summary_lines <- c(
      summary_lines,
      sprintf("cost_complexity: %s", fmt_num(pick(row, "cost_complexity"))),
      sprintf("min_n: %s", fmt_num(pick(row, "min_n"))),
      sprintf("tree_depth: %s", fmt_num(pick(row, "tree_depth")))
    )
  } else if (row$alg == "nnet") {
    summary_lines <- c(
      summary_lines,
      sprintf("hidden_units: %s", fmt_num(pick(row, "hidden_units"))),
      sprintf("penalty: %s", fmt_num(pick(row, "penalty")))
    )
  } else if (row$alg == "knn") {
    summary_lines <- c(summary_lines, sprintf("neighbors: %s", fmt_num(pick(row, "neighbors"))))
  } else if (row$alg == "svm") {
    summary_lines <- c(
      summary_lines,
      sprintf("cost: %s", fmt_num(pick(row, "cost"))),
      sprintf("rbf_sigma: %s", fmt_num(pick(row, "rbf_sigma")))
    )
  }
  summary_lines <- c(
    summary_lines,
    sprintf("CV accuracy (mean over 3 folds): %s", fmt_num(row$mean)),
    sprintf("Test-set accuracy: %s", fmt_num(row$.estimate)),
    ""
  )
}

writeLines(summary_lines, summary_path)
writeLines(summary_lines)
