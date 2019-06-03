# test alpha estimation procedure
n <- 200
ds <- c(400, 200, 100)
simulatedData <- dataSimu_group_sparse(n=n,ds=ds,
                                       dataTypes= 'GGG',
                                       noises=rep(1,3),
                                       margProb=0.1,sparse_ratio=0,
                                       SNRgc=1,SNRlc=rep(1,3),SNRd=rep(1,3))
dataSets <- simulatedData$X
alphas_est <- rep(NA,3)

opts <- list()
opts$tol_obj <- 1E-4
opts$quiet <- 1

for (i in 1:3){
  # index ith data set
  X <- dataSets[[i]]

  # add element-wise missing pattern
  full_indexes <- 1:(n*dim(X)[2])
  missing_index <- sample(full_indexes, round(0.1*length(full_indexes)))
  X[missing_index] <- NA

  # add row-wise and column-wise missing pattern
  X[sample(1:n, round(0.1*n)),] <- NA
  X[,sample(1:dim(X)[2], round(0.1*dim(X)[2]))] <- NA

  # alpha estimation procedure
  alpha_test <- alpha_estimation(X,K=3,Rs=5:20,opts=opts)
  alphas_est[i] <- alpha_test$alphas_mean
}
expect_equal(rep(1,3), alphas_est, tolerance = 0.1)





