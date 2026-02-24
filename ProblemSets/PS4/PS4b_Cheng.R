options(stringsAsFactors = FALSE)

needed <- c("sparklyr", "tidyverse")
missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

library(sparklyr)
library(tidyverse)

sc <- spark_connect(master = "local")

df1 <- as_tibble(iris)
df <- copy_to(sc, df1, overwrite = TRUE)

cat("===== class(df1) =====\n")
print(class(df1))

cat("===== class(df) =====\n")
print(class(df))

cat("===== colnames(df1) =====\n")
print(colnames(df1))

cat("===== colnames(df) =====\n")
print(colnames(df))

if (!("Sepal_Length" %in% colnames(df)) && ("Sepal.Length" %in% colnames(df))) {
  df <- df %>% rename(Sepal_Length = Sepal.Length)
}

cat("===== select Sepal_Length, Species (first 6) =====\n")
df %>% select(Sepal_Length, Species) %>% head %>% print()

cat("===== filter Sepal_Length > 5.5 (first 6) =====\n")
df %>% filter(Sepal_Length > 5.5) %>% head %>% print()

cat("===== combined filter + select (first 6) =====\n")
df %>% filter(Sepal_Length > 5.5) %>% select(Sepal_Length, Species) %>% head %>% print()

cat("===== group_by Species summarize mean and count =====\n")
df2 <- df %>%
  group_by(Species) %>%
  summarize(mean = mean(Sepal_Length), count = n())
df2 %>% head %>% print()

cat("===== arrange by Species =====\n")
tryCatch(
  {
    df2 %>% arrange(Species) %>% head %>% print()
  },
  error = function(e) {
    cat("arrange() failed:", conditionMessage(e), "\n")
  }
)

spark_disconnect(sc)
