#' alpha estimation procedure
#'
#' This function will estimate the noise level (alpha is the variation parameter)
#' of a quantitative data set using the PCA model. The rank of the PCA model
#' is selected using a missing value based cross validation procedure. The
#' details of this function can be found in \url{https://arxiv.org/abs/1902.06241}.
#'
#' @param X a quantitative data set
#' @param K K-fold cross validation
#' @param Rs the searching range of the number of components
#' @param opts a list contains the setting for the algorithm. \itemize{
#' \item tol_obj: tolerance for relative change of hist_obj, default:1E-6;
#' \item maxit: max number of iterations, default: 1000;
#' }
#'
#' @return This function returns a list contains \itemize{
#' \item alphas_mean: the mean of the K estimated alphas
#' \item alphas_std: the std of the K estimated alphas
#' \item R_CV: the number of PCs with minimum cross validation error
#' \item cvErrors: K-fold cross validation errors
#' }
#'
#' @import RSpectra
#' @importFrom stats sd
#'
#' @examples
#' \dontrun{alpha_estimation(X,K=3,Rs = 1:15,opts=list())}
#'
#' @export
alpha_estimation <- function(X, K = 3, Rs = 1:15, opts = list()) {
    # check if the whole row or whole column is missing index out the rows and columns not fully missing
    W <- 1 - is.na(X)
    X <- X[rowSums(W) > 0, colSums(W) > 0]
    
    # first center the data sets
    m <- dim(X)[1]
    n <- dim(X)[2]
    X <- scale(X, center = TRUE, scale = FALSE)  # NAs are omitted
    
    # parameters
    W <- 1 - is.na(X)
    mn_nonNaN <- sum(W)
    
    # model selection
    ratio_mis = 0.1
    cvErrors <- svd_CV(X, K, Rs, ratio_mis, opts)
    cvErrors <- cvErrors[, -(K + 1)]
    
    # select the number of components
    R_CV <- Rs[apply(cvErrors, 2, which.min)]
    
    # estimate the noise level
    alphas_CV <- rep(NA, length(R_CV))
    
    # first do a SVD on X to accelerate computation
    R_CV_max <- max(R_CV)
    if (all(!is.na(X))) {
        svd_tmp <- RSpectra::svds(X, R_CV_max, nu = R_CV_max, nv = R_CV_max)
        U <- svd_tmp$u
        S <- diag(svd_tmp$d)
        V <- svd_tmp$v
    } else {
        svd_tmp <- svd_mis(X, R_CV_max, opts)
        U <- svd_tmp$U
        S <- diag(as.vector(svd_tmp$S))
        V <- svd_tmp$V
    }
    
    for (i in 1:length(R_CV)) {
        R_CV_tmp <- R_CV[i]
        Z_hat <- U[, 1:R_CV_tmp] %*% S[1:R_CV_tmp, 1:R_CV_tmp] %*% t(V[, 1:R_CV_tmp])
        DF <- mn_nonNaN - (m + n) * R_CV_tmp
        X_nonNaN <- X
        X_nonNaN[is.na(X)] <- 0
        Z_hat[is.na(X)] <- 0
        sigmaSqure <- (1/DF) * norm(X_nonNaN - Z_hat, "F")^2
        alphas_CV[i] <- sigmaSqure
    }
    alphas_mean <- mean(alphas_CV)
    alphas_std <- stats::sd(alphas_CV)
    
    result <- list()
    result$alphas_mean <- alphas_mean
    result$alphas_std <- alphas_std
    result$alphas_CV <- alphas_CV
    result$R_CV <- R_CV
    result$cvErrors <- cvErrors
    return(result)
}
