#ifndef MODULES_PESCA_ALGORITHM_H
#define MODULES_PESCA_ALGORITHM_H

#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <math.h>
#include <algorithm>
#include "concave_penalties.h"

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[RcPP::plugins("cpp11")]]

// map function names in string format to the corresponding function pointer
std::map<std::string, std::function<arma::vec(arma::vec,double,double)> > concave_funs;

// indexes of the ith data set
arma::uvec index_Xi(int i, Rcpp::IntegerVector ds);

/*
#' Updating loading matrix B when conave L2 norm penalty is used
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
arma::mat update_B_L2(const arma::mat &JHk,
                      const arma::mat &A,
                      const arma::mat &B0,
                      const arma::mat &Sigmas0,
                      const Rcpp::IntegerVector &d,
                      std::string fun_name,
                      const Rcpp::NumericVector &alphas,
                      const Rcpp::NumericVector &rhos,
                      const Rcpp::NumericVector &lambdas,
                      double gamma);
					  
/*
#' Group-wise conave L2 norm penalty
#'
#' This is an intermediate step of the algorithm for fitting pESCA model. The
#' details of this function can be found in ref thesis.
#'
#' @param B_i The loading matrix for the ith data set
#' @param fun_concave A string indicates the used concave function
#' @param gamma The hyper-parameter of the concave penalty
#' @param R The number of PCs
#'
#' @return This function returns the value of the
#' group-wise conave L2 norm penalty for the pESCA model
#'
#' @examples
#' \dontrun{
#' concave_L2(B_i, fun_concave, gamma, R)
#' }
*/
Rcpp::List concave_L2(const arma::mat &B_i,
                      const std::string &fun_name,
                      double gamma);

//' Updating loading matrix B with conave L1 norm penalty
//'
//' This is an intermediate step of the algorithm for fitting pESCA model. The
//' details of this function can be found in ref thesis.
//'
//' @inheritParams update_B_L2
//'
//' @return This function returns the updated loading matrix B.
//'
//' @examples
//' \dontrun{
//' B <- update_B_L1(JHk,A,B0,Sigmas0,d,
//'                    fun_concave,alphas,rhos,lambdas,gamma)
//' }
arma::mat update_B_L1(const arma::mat &JHk,
                      const arma::mat &A,
                      const arma::mat &B0,
                      const arma::mat &Sigmas0,
                      const Rcpp::IntegerVector &d,
                      std::string fun_name,
                      const Rcpp::NumericVector &alphas,
                      const Rcpp::NumericVector &rhos,
                      const Rcpp::NumericVector &lambdas,
                      double gamma);					  

//' Group-wise conave L1 norm penalty
//'
//' This is an intermediate step of the algorithm for fitting pESCA model. The
//' details of this function can be found in ref thesis.
//'
//' @inheritParams penalty_concave_L2
//'
//' @return This function returns the value of the
//' group-wise conave L1 norm penalty for the pESCA model
//'
//' @examples
//' \dontrun{
//' penalty_concave_L1(B_i, fun_concave, gamma, R)
//' }
Rcpp::List concave_L1(const arma::mat &B_i,
                      const std::string &fun_name,
                      double gamma);
					  
//' Updating loading matrix B with the composite concave penalty
//'
//' This is an intermediate step of the algorithm for fitting pESCA model. The
//' details of this function can be found in ref thesis.
//'
//' @inheritParams update_B_L2
//'
//' @return This function returns the updated loading matrix B.
//'
//' @examples
//' \dontrun{
//' B <- update_B_composite(JHk,A,B0,Sigmas0,d,
//'                    fun_concave,alphas,rhos,lambdas,gamma)
//' }
arma::mat update_B_composite(const arma::mat &JHk,
                      const arma::mat &A,
                      const arma::mat &B0,
                      const arma::mat &Sigmas0,
                      const Rcpp::IntegerVector &d,
                      std::string fun_name,
                      const Rcpp::NumericVector &alphas,
                      const Rcpp::NumericVector &rhos,
                      const Rcpp::NumericVector &lambdas,
                      double gamma);					  

//' Composition of group-wise and element-wise conave penalty
//'
//' This is an intermediate step of the algorithm for fitting pESCA model. The
//' details of this function can be found in ref thesis.
//'
//' @inheritParams penalty_concave_L2
//'
//' @return This function returns the value of the composition
//' of group-wise and element-wise conave penalty for the pESCA model
//'
//' @examples
//' \dontrun{
//' concave_composite(B_i, fun_concave, gamma, R)
//' }
Rcpp::List concave_composite(const arma::mat &B_i,
                             const std::string &fun_name,
                             double gamma);							

//' Updating loading matrix B with the element-wise concave penalty
//'
//' This is an intermediate step of the algorithm for fitting pESCA model. The
//' details of this function can be found in ref thesis.
//'
//' @inheritParams update_B_L2
//'
//' @return This function returns the updated loading matrix B.
//'
//' @examples
//' \dontrun{
//' B <- update_B_element(JHk,A,B0,Sigmas0,d,
//'                    fun_concave,alphas,rhos,lambdas,gamma)
//' }
arma::mat update_B_elment(const arma::mat &JHk,
                          const arma::mat &A,
                          const arma::mat &B0,
                          const arma::mat &Sigmas0,
                          const Rcpp::IntegerVector &d,
                          std::string fun_name,
                          const Rcpp::NumericVector &alphas,
                          const Rcpp::NumericVector &rhos,
                          const Rcpp::NumericVector &lambdas,
                          double gamma);
						  
//' Element-wise conave penalty
//'
//' This is an intermediate step of the algorithm for fitting pESCA model. The
//' details of this function can be found in ref thesis.
//'
//' @inheritParams penalty_concave_L2
//'
//' @return This function returns the value of the
//' element-wise conave penalty for the pESCA model
//'
//' @examples
//' \dontrun{
//' penalty_concave_element(B_i, fun_concave, gamma, R)
//' }
Rcpp::List concave_element(const arma::mat &B_i,
                           const std::string &fun_name,
                           double gamma);
						   
#endif