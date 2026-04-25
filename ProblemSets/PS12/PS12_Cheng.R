###############################################################################
## Econ 5253 -- Spring 2026 -- Problem Set 12
## Author: Cheng
## Description:
##   - Load wages12.csv
##   - Summary statistics
##   - Three imputation methods for missing log wages
##       (1) Listwise deletion (complete case)
##       (2) Mean imputation
##       (3) Heckman sample-selection (Heckit, 2-step)
##   - Probit model for union job choice
##   - Counterfactual: zero out coefficients on married & kids
###############################################################################

suppressPackageStartupMessages({
    library(sampleSelection)
    library(tidyverse)
    library(modelsummary)
})

## ---- 4. Load data ----------------------------------------------------------
wagedata <- read_csv("wages12.csv", show_col_types = FALSE)

cat("\n--- Raw data structure ---\n")
print(dim(wagedata))
print(head(wagedata))

## ---- 5. Format college, married, union as factors --------------------------
wagedata <- wagedata %>%
    mutate(college = factor(college, levels = c(0, 1), labels = c("No", "Yes")),
           married = factor(married, levels = c(0, 1), labels = c("No", "Yes")),
           union   = factor(union,   levels = c(0, 1), labels = c("No", "Yes")))

## ---- 6. Summary table ------------------------------------------------------
cat("\n--- Summary statistics ---\n")
datasummary_skim(wagedata,
                 output = "summary_table.tex",
                 title  = "Summary Statistics for wages12 Data",
                 fmt    = 3)
datasummary_skim(wagedata, output = "markdown")

missing_rate <- mean(is.na(wagedata$logwage))
cat(sprintf("\nMissing logwage rate: %.4f (%d / %d obs)\n",
            missing_rate, sum(is.na(wagedata$logwage)), nrow(wagedata)))

## ---- 7. Three imputation methods -------------------------------------------

## (a) Listwise deletion (complete cases) ------------------------------------
est_complete <- lm(logwage ~ hgc + union + college + exper + I(exper^2),
                   data = wagedata)
cat("\n--- (a) Listwise deletion ---\n")
print(summary(est_complete))

## (b) Mean imputation -------------------------------------------------------
mean_logwage <- mean(wagedata$logwage, na.rm = TRUE)
wagedata_mean <- wagedata %>%
    mutate(logwage = if_else(is.na(logwage), mean_logwage, logwage))

est_mean <- lm(logwage ~ hgc + union + college + exper + I(exper^2),
               data = wagedata_mean)
cat("\n--- (b) Mean imputation ---\n")
print(summary(est_mean))

## (c) Heckman sample-selection model ---------------------------------------
wagedata_heck <- wagedata %>%
    mutate(valid   = as.numeric(!is.na(logwage)),
           logwage = if_else(is.na(logwage), 0, logwage))

est_heckit <- selection(
    selection = valid ~ hgc + union + college + exper + married + kids,
    outcome   = logwage ~ hgc + union + college + exper + I(exper^2),
    data      = wagedata_heck,
    method    = "2step")

cat("\n--- (c) Heckman 2-step selection model ---\n")
print(summary(est_heckit))

## ---- Combined regression table --------------------------------------------
## The Heckit object has two equations (selection + outcome) with coefficients
## of the same name, so we extract only the outcome equation and wrap it in
## a `modelsummary_list` so it can be combined with the two lm models.
heck_sum  <- summary(est_heckit)
heck_out  <- heck_sum$estimate[heck_sum$param$index$outcome, , drop = FALSE]

heckit_ms <- list(
    tidy = data.frame(
        term      = rownames(heck_out),
        estimate  = heck_out[, "Estimate"],
        std.error = heck_out[, "Std. Error"],
        statistic = heck_out[, "t value"],
        p.value   = heck_out[, "Pr(>|t|)"],
        row.names = NULL),
    glance = data.frame(
        nobs       = unname(heck_sum$param$nObs),
        n.observed = unname(heck_sum$param$N1),
        n.censored = unname(heck_sum$param$N0)))
class(heckit_ms) <- "modelsummary_list"

models <- list("Listwise deletion" = est_complete,
               "Mean imputation"   = est_mean,
               "Heckit (outcome eq)" = heckit_ms)

modelsummary(models,
             output    = "regression_table.tex",
             stars     = TRUE,
             title     = "Returns to schooling under three missing-data treatments",
             gof_omit  = "AIC|BIC|Log.Lik|F|RMSE",
             notes     = "Heckit reports the outcome equation only.")

## Markdown copy for the console
modelsummary(models, output = "markdown",
             stars    = TRUE,
             gof_omit = "AIC|BIC|Log.Lik|F|RMSE")

## ---- 8. Probit model of union-job preferences -----------------------------
probit_union <- glm(union ~ hgc + college + exper + married + kids,
                    family = binomial(link = "probit"),
                    data   = wagedata)

cat("\n--- Probit: union job choice ---\n")
print(summary(probit_union))

## Save probit table for the writeup
modelsummary(list("Probit: union" = probit_union),
             output = "probit_table.tex",
             stars  = TRUE,
             title  = "Probit model of union-job choice",
             gof_omit = "AIC|BIC|Log.Lik|F|RMSE")

## ---- 9. Counterfactual: zero-out married and kids coefficients ------------
wagedata$predProbit <- predict(probit_union, newdata = wagedata,
                               type = "response")
mean_orig <- mean(wagedata$predProbit, na.rm = TRUE)

probit_cfl <- probit_union
cat("\nOriginal coefficient names:\n"); print(names(probit_cfl$coefficients))

probit_cfl$coefficients["marriedYes"] <- 0
probit_cfl$coefficients["kids"]       <- 0

wagedata$predProbitCfl <- predict(probit_cfl, newdata = wagedata,
                                  type = "response")
mean_cfl <- mean(wagedata$predProbitCfl, na.rm = TRUE)

cat(sprintf("\n--- Counterfactual comparison ---\n"))
cat(sprintf("Mean predicted Pr(union) -- baseline       : %.4f\n", mean_orig))
cat(sprintf("Mean predicted Pr(union) -- counterfactual : %.4f\n", mean_cfl))
cat(sprintf("Difference (cfl - baseline)                 : %.4f\n",
            mean_cfl - mean_orig))

## Save counterfactual numbers for the writeup
writeLines(c(sprintf("Missing rate: %.4f", missing_rate),
             sprintf("beta1 listwise : %.4f", coef(est_complete)["hgc"]),
             sprintf("beta1 mean-imp : %.4f", coef(est_mean)["hgc"]),
             sprintf("beta1 heckit   : %.4f",
                     heck_out["hgc", "Estimate"]),
             sprintf("Pr(union) baseline       : %.4f", mean_orig),
             sprintf("Pr(union) counterfactual : %.4f", mean_cfl),
             sprintf("Difference (cfl - baseline): %.4f",
                     mean_cfl - mean_orig)),
           con = "results_summary.txt")

cat("\nDone.\n")
