options(stringsAsFactors = FALSE)

needed <- c("jsonlite", "tidyverse")
missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

library(jsonlite)
library(tidyverse)

url <- "https://www.vizgr.org/historical-events/search.php?format=json&begin_date=00000101&end_date=20240209&lang=en"
system(paste0('wget -O dates.json "', url, '"'))
cat("Downloaded dates.json\n")

cat("===== cat dates.json =====\n")
system("cat dates.json")

mylist <- fromJSON("dates.json")
mydf <- bind_rows(mylist$result[-1])

cat("===== class(mydf) =====\n")
print(class(mydf))

cat("===== class(mydf$date) =====\n")
print(class(mydf$date))

cat("===== head(mydf, 10) =====\n")
print(head(mydf, 10))
