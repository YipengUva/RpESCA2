#ifndef MODULES_PESCA_ALGORITHM_H
#define MODULES_PESCA_ALGORITHM_H

#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <math.h>
#include <algorithm>
#include "concave_functions.h"

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]

// map concave function and its supergradient names 
// in string format to the corresponding function pointer
std::map<std::string, 
         std::function<arma::vec(arma::vec,double,double)> > Funs;

// indexes of the ith data set
arma::uvec index_Xi(int i, Rcpp::IntegerVector ds);

//' Updating loading matrix B when conave L2 norm penalty is used
//'
//' This is an intermediate step of the algorithm for fitting pESCA model. The
//' details of this function can be found in ref thesis.
//'
//' @param JHk An output of the majorizaiotn step
//' @param A The score matrix A during k-th iteration
//' @param B0 The loading matrix B during the previous iteration
//' @param Sigmas0 The group length during the previous iteration
//' @param d A numeric vector contains the numbers of variables in different data sets
//' @param fun_concave A string indicates the used concave function
//' @param alphas The dispersion parameters of exponential dispersion families
//' @param rhos An output of the majorizaiotn step
//' @param lambdas A numeric vector indicates the values of tuning parameters for
//' each data set.
//' @param gamma The hyper-parameter of the concave penalty
//'
//' @return This function returns the updated loading matrix B.
//'
//' @examples
//' \dontrun{
//' B <- update_B_L2(JHk,A,B0,Sigmas0,d,
//'                    fun_concave,alphas,rhos,lambdas,gamma)
//' }
void update_B_L2(const arma::mat &JHk,
                 const arma::mat &A,
                 arma::mat &B,
                 const arma::mat &Sigmas,
                 const Rcpp::IntegerVector &d,
                 const std::string &fun_concave,
                 const Rcpp::NumericVector &alphas,
                 const Rcpp::NumericVector &rhos,
                 const Rcpp::NumericVector &lambdas,
                 const double &gamma);
					  
//' Group-wise conave L2 norm penalty
//'
//' This is an intermediate step of the algorithm for fitting pESCA model. The
//' details of this function can be found in ref thesis.
//'
//' @param B The loading matrix for the ith data set
//' @param Sigmas The matrix to hold the group characters
//' @param d the vector contains the number of variables in each data set
//' @param fun_concave A string indicates the used concave function
//' @param gamma The hyper-parameter of the concave penalty
//'
//' @return This function returns the value of the
//' group-wise conave L2 norm penalty for the pESCA model
//'
//' @examples
//' \dontrun{
//' concave_L2(B_i, fun_concave, gamma, R)
//' }
double penalty_L2(const arma::mat &B,
                  arma::mat &Sigmas,
                  const Rcpp::IntegerVector &d,
				  const Rcpp::NumericVector &lambdas,
                  const std::string &fun_concave,
                  const double &gamma);

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
void arma::mat update_B_L1(const arma::mat &JHk,
                           const arma::mat &A,
                           arma::mat &B,
                           const arma::mat &Sigmas,
                           const Rcpp::IntegerVector &d,
                           const std::string &fun_concave,
                           const Rcpp::NumericVector &alphas,
                           const Rcpp::NumericVector &rhos,
                           const Rcpp::NumericVector &lambdas,
                           const double &gamma);					  

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
double penalty_L1(const arma::mat &B,
                  arma::mat &Sigmas,
                  const Rcpp::IntegerVector &d,
				  const Rcpp::NumericVector &lambdas,
                  const std::string &fun_concave,
                  const double &gamma);
					  
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
void update_B_composite(const arma::mat &JHk,
                        const arma::mat &A,
                        arma::mat &B,
                        const arma::mat &Sigmas,
                        const Rcpp::IntegerVector &d,
                        const std::string &fun_concave,
                        const Rcpp::NumericVector &alphas,
                        const Rcpp::NumericVector &rhos,
                        const Rcpp::NumericVector &lambdas,
                        const double &gamma);					  

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
double penalty_composite(const arma::mat &B,
                         arma::mat &Sigmas,
                         const Rcpp::IntegerVector &d,
						 const Rcpp::NumericVector &lambdas,
                         const std::string &fun_concave,
                         const double &gamma);							

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
void update_B_elment(const arma::mat &JHk,
                     const arma::mat &A,
                     arma::mat &B,
                     const arma::mat &Sigmas,
                     const Rcpp::IntegerVector &d,
                     const std::string &fun_concave,
                     const Rcpp::NumericVector &alphas,
                     const Rcpp::NumericVector &rhos,
                     const Rcpp::NumericVector &lambdas,
                     const double &gamma);
						  
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
double penalty_element(const arma::mat &B,
                       arma::mat &Sigmas,
                       const Rcpp::IntegerVector &d,
					   const Rcpp::NumericVector &lambdas,
                       const std::string &fun_concave,
                       const double &gamma);
					   
// This is an implementaion of fast verion trace function.
// This function will compute the trace of two matrices.
double trace_fast(const arma::mat &X, const arma::mat &Y);

// variation explained ratios
Rcpp::NumericMatrix varExp_Gaussian(const arma::mat &X,
                                    const Rcpp::IntegerVector &d,
                                    const arma::rowvec &mu,
                                    const arma::mat &A,
                                    const arma::mat &B,
                                    const arma::imat &W);
						   
#endif