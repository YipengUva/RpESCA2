library(Rcpp)
library(RcppArmadillo)
library(microbenchmark)

n <- 200
ds <- c(100, 500, 100)
simulatedData <- RpESCA::dataSimu_group_sparse(n=n,ds=ds,
                                       dataTypes= 'GGG',
                                       noises=rep(1,3),
                                       margProb=0.1,sparse_ratio=0,
                                       SNRgc=1,SNRlc=rep(1,3),SNRd=rep(1,3))

dataSets <- simulatedData$X
alphas_est <- rep(NA,3)

opts <- list()
opts$tol_obj <- 1E-6
opts$quiet <- 1

for (i in 1:1){
  # index ith data set
  X <- dataSets[[i]]
  
  # add element-wise missing pattern
  full_indexes <- 1:(n*dim(X)[2])
  missing_index <- sample(full_indexes, round(0.2*length(full_indexes)))
  X[missing_index] <- NA
  
  # add row-wise and column-wise missing pattern
  X[sample(1:n, round(0.1*n)),] <- NA
  X[,sample(1:dim(X)[2], round(0.1*dim(X)[2]))] <- NA
  
  # alpha estimation procedure
  alpha_test <- alpha_estimation(X,K=3,Rs = 5:20,opts=opts)
  alphas_est[i] <- alpha_test$alphas_mean
}


## experiemnt
svd_CVC = svd_CV(X,K=3,Rs = 1:15,ratio_mis=0.1,opts=opts)

svd_CVR = RpESCA::svd_CV(X,K=3,Rs = 1:15,ratio_mis=0.1,opts=opts)

svd_CVC
svd_CVR$cvErrors

microbenchmark(svd_CV(X,K=3,Rs = 1:15,ratio_mis=0.1,opts=opts),
               RpESCA::svd_CV(X,K=3,Rs = 1:15,ratio_mis=0.1,opts=opts),
               times = 10)



A = matrix(1:10, 2,5);
A[2,4] = NA;
A[2,5] = NA;
A[1,2] = NaN;
A[1,3] = NaN;

arma_exp(A)




R = 5;
svdmisC = svd_mis(X,R,opts)
svdmisC$iter
svdmisC$diagnose


svdmisR = RpESCA::svd_mis(X,R,opts)
svdmisR$iter
svdmisR$diagnose

all.equal(svdmisC$U, svdmisR$U)
all.equal(svdmisC$V, svdmisR$V)
svdmisC$S
svdmisR$S
diag(svdmisR$S)

microbenchmark(svd_mis(X,R,opts),
               RpESCA::svd_mis(X,R,opts))

