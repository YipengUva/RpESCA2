#' Index generating function
#'
#' This function will generate a series of indexes used to index out
#' the variables of the ith data set when we only know the number of
#' variables
#'
#' @param i index for the ith data set.
#' @param ds a vector contains the number of variables for all
#' the data sets.
#'
#' @return A series of indexes for the variables in the ith data set
#'
#' @examples
#' \dontrun{index_Xi(2, c(400,200,100))}
index_Xi <- function(i, ds) {
  if (i == 1) {
    columns_Xi <- 1:ds[1]
  } else {
    columns_Xi <- (sum(ds[1:(i - 1)]) + 1):sum(ds[1:i])
  }
  columns_Xi
}

#' Penalized exponential family simultaneous component analysis (pESCA) model
#'
#' This is the main function for construncting a pESCA model on multiple data
#' sets. The potential different data types in these data sets are tackled by
#' the assumption of exponential family distribution. Gaussian for quantitative
#' data, Bernoulli for binary data and Poisson for count data. Although the option
#' for count data using Poisson distribution is included in the algorithm, we recommend
#' to do variance stabilizing transformation on the count data, such as Next-Gen
#' sequencing data, and then use the transformed data as quantitative data sets. The
#' details of the developed algorithm can be found in \url{https://arxiv.org/abs/1902.06241}.
#'
#' @param dataSets a list contains multiple matrices with same number of rows.
#' Each matrix (\code{samples * variables}) indicates a data set.
#' @param dataTypes a string indicates the data types of the multiple data sets.
#' @param lambdas a numeric vector indicates the values of tuning parameters for
#' each data set.
#' @param penalty The name of the penalty you want to used. \itemize{
#' \item "L2": group-wise concave L2 norm penalty;
#' \item "L1": group-wise concave L1 norm penalty;
#' \item "element": element-wise concave penalty;
#' \item "composite": the composition of group- and element-wise penalty.
#' }
#' @param fun_concave a string indicates the used concave function. Three options
#' are included in the algorithm. \itemize{
#' \item "gdp": GDP penalty;
#' \item "lq": Lq penalty;
#' \item "scad": SCAD penalty.
#' }
#' @param opts a list contains the options of the algorithms. \itemize{
#' \item tol_obj: tolerance for relative change of joint loss function, default:1E-6;
#' \item maxit: max number of iterations, default: 1000;
#' \item gamma: hyper-parameter of the concave penalty, default: 1;
#' \item R: the initial number of PCs, default: 0.5 \code{0.5*min(I,J)};
#' \item rand_start: initilization method, random (1), SCA(0), default: 0;
#' \item alphas: dispersion parameters of exponential dispersion families, default: 1.
#' \item thr_path: the option to generate thresholding path, default: 0;
#' \item quiet: quiet==1, not show the progress when running the algorithm, default: 0.
#' }
#'
#' @return This function returns a list contains the results of a pESCA mdoel. \itemize{
#' \item mu: offset term;
#' \item A: score matrix;
#' \item B: loading matrix;
#' \item S: group sparse pattern of B;
#' \item varExpTotals: total variation explained of each data set and the full data set;
#' \item varExpPCs: variation explained of each data set and each component;
#' \item Sigmas: the group length (the definition depends on the used input type). Only meaningful
#' for group-wise sparisty;
#' \item iter: number of iterations;
#' \item diagnose$hist_objs: the value of loss function pluse penalty at each iteration;
#' \item diagnose$f_objs: the value of loss function at each iteration;
#' \item diagnose$g_objs: the value of penalty function at each iteration;
#' \item diagnose$rel_objs: relative change of diagnose$hist_obj at each iteration;
#' \item diagnose$rel_Thetas: relative change of Theta at each iteration.
#' }
#'
#' @importFrom RSpectra svds
#'
#' @examples
#' \dontrun{
#' # Suppose we have three data sets X1, X2, X3
#' # They are quantitative, quantitative and binary matrices
#' pESCA(dataSets = list(X1, X2, X3),
#'               dataTypes = 'GGB',
#'               lambdas = c(20, 20, 10),
#'               penalty = 'L2',
#'               fun_concave = 'gdp',
#'               opts = list())
#' }
#'
#' @export
pESCA <- function(dataSets, dataTypes,
                  lambdas, penalty = 'L2', 
                  fun_concave = 'gdp', opts = list()){
  
  # check if data sets are in a list
  stopifnot(class(dataSets) == "list") 
  
  # check if the used penalties are included in the algorithm
  possible_penalties <- c("L2", "L1", "composite", "element")
  if ( !(penalty %in% possible_penalties) )
    stop("The used penalty is not included in the algorithm")
  
  # check if the used concave function are included in the algorithm
  possible_funs <- c("gdp", "lq", "scad")
  if ( !(fun_concave %in% possible_funs) )
    stop("The used concave function is not included in the algorithm")
  
  # check if the numer of data sets are equal to the number of lambdas
  if ( !(length(dataSets) == length(lambdas) ) )
    stop("the number of data sets are not equal to the number of lambdas")
  
  # check if the used data Types are included in the algorithm
  if(length(dataTypes) == 1) dataTypes <- unlist(strsplit(dataTypes, split=""))
  if(length(dataTypes) != length(dataSets) )
    stop("The number of specified dataTypes are not equal to the number of data sets")
  
  # number of data sets, size of each data set
  nDataSets <- length(dataSets) # number of data sets
  n <- rep(0,nDataSets)  # number of samples
  d <- rep(0,nDataSets)  # numbers of variables in different data sets
  for(i in 1:nDataSets){
    n[i] <- dim(dataSets[[i]])[1]
    d[i] <- dim(dataSets[[i]])[2]
  }
  if(length(unique(n)) != 1)
    stop("multiple data sets have unequal sample size")
  n <- n[1]
  sumd <- sum(d) # total number of variables

  # form full data set X, X = [X1,...Xl,...XL]
  # test dataTypes
  X <- matrix(data=NA, nrow=n, ncol=sumd)
  test_dataTypes <- rep('G',nDataSets)

  for(i in 1:nDataSets){
    columns_Xi <- index_Xi(i, d)
    X_i <- as.matrix(dataSets[[i]])
    X[,columns_Xi] <- X_i
    
    if (max(X_i, na.rm = TRUE) == 1 & min(X_i, na.rm = TRUE) == 0){
      test_dataTypes[i] <- 'B'
    }
  }
  if(!all(dataTypes == test_dataTypes))
    warning("The specified data types may not correct")

  # using algorithm writtn in C++ to do the computation
  result <- pESCA_C(X, d, dataTypes, lambdas, penalty, fun_concave, opts)
  
  # output
  return(result)
}
