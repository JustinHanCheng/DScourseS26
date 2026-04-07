library(tidyverse)

test <- function() {
set.seed(100)
N <- 1000000
K <- 10
sigma <- 0.5
X <- matrix(15*runif(N*K),N,K)
#X[,1] <- 1 # first column of X should be all ones
eps <- rnorm(N,mean=0,sd=sigma)
betaTrue <- c(1.5, as.vector(runif(K)))
Y <- betaTrue[1] + X%*%betaTrue[2:length(betaTrue)] + eps
estimates <- lm(Y~X)
print(summary(estimates))
print(betaTrue)
}

test()
