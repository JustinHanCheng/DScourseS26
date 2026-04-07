library(tidyverse)
B <- 1000
N <- 10000
K <- 10
betaTrue <- c(1.5, as.vector(runif(K)))
sigma <- 0.5
X <- matrix(15*runif(N*K),N,K)
set.seed(100)

test <- function() {
  #X[,1] <- 1 # first column of X should be all ones
  eps <- rnorm(N,mean=0,sd=sigma)
  Y <- betaTrue[1] + X%*%betaTrue[2:length(betaTrue)] + eps
  estimates <- lm(Y~X)
  #browser()
  #print(summary(estimates))
  #print(betaTrue)
  return(estimates$coef)
}

output = matrix(nrow=B, ncol=K+1)
for (b in 1:B) {
  output[b,] <- test()
}

