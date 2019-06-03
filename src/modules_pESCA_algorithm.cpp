#include <iostream>
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[RcPP::plugins("cpp11")]]

using namespace arma;

/*
#' Updating loading matrix B with conave L2 norm penalty
#'
#' This is an intermediate step of the algorithm for fitting pESCA model. The
#' details of this function can be found in ref thesis.
#'
#' @param JHk An output of the majorizaiotn step
#' @param A The score matrix A during k-th iteration
#' @param B0 The loading matrix B during the previous iteration
#' @param Sigmas0 The group length during the previous iteration
#' @param d A numeric vector contains the numbers of variables in different data sets
#' @param fun_concave A string indicates the used concave function
#' @param alphas The dispersion parameters of exponential dispersion families
#' @param rhos An output of the majorizaiotn step
#' @param lambdas A numeric vector indicates the values of tuning parameters for
#' each data set.
#' @param gamma The hyper-parameter of the concave penalty
#'
#' @return This function returns the updated loading matrix B.
#'
#' @examples
#' \dontrun{
#' B <- update_B_L2(JHk,A,B0,Sigmas0,d,
#'                    fun_concave,alphas,rhos,lambdas,gamma)
#' }
*/




