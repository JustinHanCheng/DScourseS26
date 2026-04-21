if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("RDRToolbox")
library(RDRToolbox)
library(magrittr)
library(mclust)

# Set up the data we'll be using
X <- iris[,-5]
Y <- iris[, 5]


#-------------------------------
# k-means using the iris dataset
#-------------------------------
# call the k-means function in base R
clusters <- X %>% kmeans(3)
# equivalently: clusters <- kmeans(X,3)

# print results
print(clusters)

# compare k-means classes with actual classes
print(table(Y,clusters$cluster))


#-------------------------------
# EM using the iris dataset
#-------------------------------
# call the EM clustering function from the mclust package
clusters <- X %>% Mclust(G=3)
# equivalently: clusters <- Mclust(X,G=3)

# list inferred probabilities of Pr(Y=y)
print(clusters$parameters$pro)

# list frequencies of each species in iris data
print(table(Y))

# list average of X's conditional on being in class y
print(clusters$parameters$mean)

# list posterior class probabilities for each observation in our training data
head(clusters$z)

# compare EM classes with actual classes
print(table(Y,clusters$classification))


#-------------------------------
# Principal components analysis
#-------------------------------
# computing eigen values
cormat <- X %>% cor(.)
print(cormat)
print(eigen(cormat)$values)

# compute eigen vectors

v <- eigen(cormat)$vectors
print(crossprod(v)) # should be identity matrix

# do PCA using base R function prcomp
prin.comp <- prcomp(X, scale. = T)

# show eigen vectors (will be same as v)
print(prin.comp$rotation)

# show eigen values
print(prin.comp$sdev^2)

# which components explain the most variance?
prop.var.explained <- prin.comp$sdev^2/sum(prin.comp$sdev^2)
print(prop.var.explained)

# keep only the first three components (since these explain 99.5% of the variance)
reducedX <- prin.comp$x[,seq(1,3)]
reducedXlook <- prin.comp$x[,seq(1,3)] %>% as.data.frame()
cormat.new <- reducedXlook %>% cor(.)
print(cormat.new)

## try EM clustering on reduced data ... can we still get the same answer?
#-------------------------------
clustersDR <- reducedX %>% Mclust(G=3)
# equivalently: clusters <- Mclust(X,G=3)

# list inferred probabilities of Pr(Y=y)
print(clustersDR$parameters$pro)

# list frequencies of each species in iris data
print(table(Y))

# list average of X's conditional on being in class y
print(clustersDR$parameters$mean)

# list posterior class probabilities for each observation in our training data
head(clustersDR$z)

# compare EM classes with actual classes
print(table(Y,clustersDR$classification))


## try k-means on reduced data ... how does it perform?
#-------------------------------
clustersKDR <- reducedX %>% kmeans(3)
# equivalently: clusters <- kmeans(X,3)

# print results
print(clustersKDR)

# compare k-means classes with actual classes
print(table(Y,clustersKDR$cluster))


## Isomap
isomapped <- Isomap(data=as.matrix(X), dims=3, k=5)
XreducedNL <- isomapped$dim3


