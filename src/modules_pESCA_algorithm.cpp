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
         std::function<arma::vec(arma::vec,double,double)> > Funs = 
  {{"gdp", gdp},
  {"gdp_sg", gdp_sg},
  {"lq", lq},
  {"lq_sg", lq_sg},
  {"scad", scad},
  {"scad_sg", scad_sg}
  };

// indexes of the ith data set
arma::uvec index_Xi(int i, Rcpp::IntegerVector ds){
  arma::uvec indexes(2);
  if(i == 0){
    indexes(0) = 0;
    indexes(1) = (ds[i] - 1);
  }else{
    indexes(0) = Rcpp::sum(ds[Rcpp::seq(0, i-1)]);
    indexes(1) = Rcpp::sum(ds[Rcpp::seq(0, i)]) - 1;
  }
  
  return indexes;
}

// Updating loading matrix B when conave L2 norm penalty is used
void update_B_L2(const arma::mat &JHk,
                 const arma::mat &A,
                 arma::mat &B,
                 const arma::mat &Sigmas,
                 const Rcpp::IntegerVector &d,
                 const std::string &fun_concave,
                 const Rcpp::NumericVector &alphas,
                 const Rcpp::NumericVector &rhos,
                 const Rcpp::NumericVector &lambdas,
                 const double &gamma){
  
  // super gradient of penalty function
  std::string hfun_sg = fun_concave + "_sg";
  
  int nDataSets = d.length();
  for(int i = 0; i < nDataSets; ++i){
    arma::uvec indexes = index_Xi(i, d);
    const arma::mat &JHk_i = JHk.cols(indexes(0), indexes(1));
    arma::mat JHkitA = JHk_i.t() * A;
    double alpha_i = alphas(i);
    double rho_i = rhos(i);
    double weight_i = std::sqrt(d(i)); // weight for L2 norm 
    double lambda_i = lambdas(i) * weight_i * alpha_i / rho_i;
    
    // form weights of the penalty according to previous sigma0_ir
    const arma::vec &sigma0_i = Sigmas.row(i);
    arma::vec omega_i  = Funs[hfun_sg](sigma0_i, gamma, 1); // weights
    
    int R = A.n_cols;
    for(int r = 0; r < R; ++r){
      // proximal operator of L2 norm
      const arma::vec &JHkitA_r = JHkitA.col(r);
      double lambda_ir = lambda_i * omega_i(r);
      double JHkitA_r_norm = arma::norm(JHkitA_r, 2);
      
      arma::vec B_ir = (1-(lambda_ir/JHkitA_r_norm)) * JHkitA_r;
      B_ir.elem(find(B_ir < 0)).zeros();
      B(arma::span(indexes(0), indexes(1)), arma::span(r, r)) = B_ir;
    }
  }
}


// Updating loading matrix B when conave L1 norm penalty is used
void arma::mat update_B_L1(const arma::mat &JHk,
                           const arma::mat &A,
                           arma::mat &B,
                           const arma::mat &Sigmas,
                           const Rcpp::IntegerVector &d,
                           const std::string &fun_concave,
                           const Rcpp::NumericVector &alphas,
                           const Rcpp::NumericVector &rhos,
                           const Rcpp::NumericVector &lambdas,
                           const double &gamma){

  // super gradient of penalty function
  std::string hfun_sg = fun_concave + "_sg";
  
  int nDataSets = d.length();
  for(int i = 0; i < nDataSets; ++i){
    arma::uvec indexes = index_Xi(i, d);
    const arma::mat &JHk_i = JHk.cols(indexes(0), indexes(1));
    arma::mat JHkitA = JHk_i.t() * A;
    double alpha_i = alphas(i);
    double rho_i = rhos(i);
    double weight_i = d(i); // weight for L1 norm 
    double lambda_i = lambdas(i) * weight_i * alpha_i / rho_i;
    
    // form weights of the penalty according to previous sigma0_ir
    const arma::vec &sigma0_i = Sigmas.row(i);
    arma::vec omega_i  = Funs[hfun_sg](sigma0_i, gamma, 1); // weights
    
    int R = A.n_cols;
    for(int r = 0; r < R; ++r){
      // proximal operator of L1 norm
      const arma::vec &JHkitA_r = JHkitA.col(r);
      double lambda_ir = lambda_i * omega_i(r);
      
      arma::vec B_ir_pre = arma::abs(JHkitA_r) - lambda_ir;
      B_ir_pre.elem(find(B_ir_pre < 0)).zeros();
      arma::vec B_ir = arma::sign(JHkitA_r) * B_ir_pre;
      B(arma::span(indexes(0), indexes(1)), arma::span(r, r)) = B_ir;
    }
  }
}


// Updating loading matrix B with the composite concave penalty
void update_B_composite(const arma::mat &JHk,
                        const arma::mat &A,
                        arma::mat &B,
                        const arma::mat &Sigmas,
                        const Rcpp::IntegerVector &d,
                        const std::string &fun_concave,
                        const Rcpp::NumericVector &alphas,
                        const Rcpp::NumericVector &rhos,
                        const Rcpp::NumericVector &lambdas,
                        const double &gamma){
  
  // super gradient of penalty function
  std::string hfun_sg = fun_concave + "_sg";
  
  int nDataSets = d.length();
  for(int i = 0; i < nDataSets; ++i){
    arma::uvec indexes = index_Xi(i, d);
    const arma::mat &JHk_i = JHk.cols(indexes(0), indexes(1));
    arma::mat JHkitA = JHk_i.t() * A;
    double alpha_i = alphas(i);
    double rho_i = rhos(i);
    double weight_i = d(i); // weight for composite concave function
    double lambda_i = lambdas(i) * weight_i * alpha_i / rho_i;
    
    // form weights of the penalty according to previous sigma0_ir
    // weights of the outer layer
    const arma::vec &sigma0_i = Sigmas.row(i);
    arma::vec omega_i_outer = Funs[hfun_sg](sigma0_i, gamma, 1); 
    
    int R = A.n_cols;
    for(int r = 0; r < R; ++r){
      // weights of the inner layer
      arma::vec sigma0_ir_vec = arma::abs(B(arma::span(indexes(0),indexes(1)), arma::span(r, r)));
      arma::vec omega_ir_inner = Funs[hfun_sg](sigma0_ir_vec, gamma, 1);
      
      // total weights
      arma::vec omega_ir_vec = omega_i_outer(r) * omega_ir_inner;
      
      // proximal operator of L1 norm
      const arma::vec &JHkitA_r = JHkitA.col(r);
      arma::vec lambda_ir_vec = lambda_i * omega_ir_vec;
      
      arma::vec B_ir_pre = arma::abs(JHkitA_r) - lambda_ir_vec;
      B_ir_pre.elem(find(B_ir_pre < 0)).zeros();
      arma::vec B_ir = arma::sign(JHkitA_r) * B_ir_pre;
      B(arma::span(indexes(0), indexes(1)),arma::span(r, r)) = B_ir;
    }
  }
}


// Updating loading matrix B with the element-wise concave penalty
void update_B_elment(const arma::mat &JHk,
                     const arma::mat &A,
                     arma::mat &B,
                     const arma::mat &Sigmas,
                     const Rcpp::IntegerVector &d,
                     const std::string &fun_concave,
                     const Rcpp::NumericVector &alphas,
                     const Rcpp::NumericVector &rhos,
                     const Rcpp::NumericVector &lambdas,
                     const double &gamma){
  
  // super gradient of penalty function
  std::string hfun_sg = fun_concave + "_sg";
  
  int nDataSets = d.length();
  for(int i = 0; i < nDataSets; ++i){
    arma::uvec indexes = index_Xi(i, d);
    const arma::mat &JHk_i = JHk.cols(indexes(0), indexes(1));
    arma::mat JHkitA = JHk_i.t() * A;
    double alpha_i = alphas(i);
    double rho_i = rhos(i);
    double weight_i = 1; // weight element-wise penalty 
    double lambda_i = lambdas(i) * weight_i * alpha_i / rho_i;
    
    int R = A.n_cols;
    for(int r = 0; r < R; ++r){
      // form weights of the penalty according to previous sigma0_ir
      const arma::vec &sigma0_ir_vec = arma::abs(B(arma::span(indexes(0), indexes(1)), arma::span(r, r)));
      arma::vec omega_ir_vec  = Funs[hfun_sg](sigma0_ir_vec, gamma, 1);
      
      // proximal operator of L1 norm
      const arma::vec &JHkitA_r = JHkitA.col(r);
      arma::vec lambda_ir_vec = lambda_i * omega_ir_vec;
      
      arma::vec B_ir_pre = arma::abs(JHkitA_r) - lambda_ir_vec;
      B_ir_pre.elem(find(B_ir_pre < 0)).zeros();
      arma::vec B_ir = arma::sign(JHkitA_r) * B_ir_pre;
      B(arma::span(indexes(0), indexes(1)), arma::span(r, r)) = B_ir;
    }
  }
}

// Group-wise conave L2 norm penalty
double penalty_L2(const arma::mat &B,
                  arma::mat &Sigmas,
                  const Rcpp::IntegerVector &d,
				  const Rcpp::NumericVector &lambdas,
                  const std::string &fun_concave,
                  const double &gamma){

  // name of concave function
  std::string hfun = fun_concave; 
  
  double out = {0};
  
  int nDataSets = d.length();
  int R = B.n_cols;
  for(int i = 0; i < nDataSets; ++i){
    
    // index for the ith data set
    arma::uvec indexes = index_Xi(i, d);

    // weight for the ith data set
	double lambda_i = lambdas(i);
    double weight_i = std::sqrt(d(i)); // weight when L2 norm is used
    
    arma::vec sigmas(R);
    for(int r = 0; r < R; ++r){
      const arma::vec &B_ir = B(arma::span(indexes(0), indexes(1)), arma::span(r, r));
      sigmas(r) = arma::norm(B_ir, 2); // sigma_{lr} = ||b_{lr}||_2
    }
    
    out +=  lambda_i * weight_i * arma::sum(Funs[hfun](sigmas, gamma, 1));
    Sigmas.row(i) = sigmas.t();
  }
    
  return out;  
}


// Group-wise conave L1 norm penalty
double penalty_L1(const arma::mat &B,
                  arma::mat &Sigmas,
                  const Rcpp::IntegerVector &d,
				  const Rcpp::NumericVector &lambdas,
                  const std::string &fun_concave,
                  const double &gamma){
  
  // name of concave function
  std::string hfun = fun_concave; 
  
  double out = {0};
  
  int nDataSets = d.length();
  int R = B.n_cols;
  for(int i = 0; i < nDataSets; ++i){
    // index for the ith data set	
    arma::uvec indexes = index_Xi(i, d);
    
    // weight for the ith data set
	double lambda_i = lambdas(i);
    int weight_i = B_i.n_rows; // weight when L1 norm is used
    
    arma::vec sigmas(R);
    for(int r = 0; r < R; ++r){
      const arma::vec &B_ir = B(arma::span(indexes(0), indexes(1)), arma::span(r, r));
      sigmas(r) = arma::sum(arma::abs(B_ir)); // sigma_{lr} = ||b_{lr}||_1
    }
    
    out += lambda_i * weight_i * arma::sum(Funs[hfun](sigmas, gamma, 1));
    Sigmas.row(i) = sigmas.t();
  }
  
  return out;
}

// the Composition of group-wise and element-wise conave penalty
double penalty_composite(const arma::mat &B,
                         arma::mat &Sigmas,
                         const Rcpp::IntegerVector &d,
						 const Rcpp::NumericVector &lambdas,
                         const std::string &fun_concave,
                         const double &gamma){

  // name of concave function
  std::string hfun = fun_concave;
  
  double out = {0};
  
  int nDataSets = d.length();
  int R = B.n_cols;
  for(int i = 0; i < nDataSets; ++i){
    
    // index for the ith data set
    arma::uvec indexes = index_Xi(i, d);
    
    // weight for the ith data set
	double lambda_i = lambdas(i);
    int weight_i = B_i.n_rows; // weight when composite L1 norm is used
    
    arma::vec sigmas(R);
    for(int r = 0; r < R; ++r){
      const arma::vec &B_ir = B(arma::span(indexes(0), indexes(1)), arma::span(r, r));
      sigmas(r) = arma::sum(Funs[hfun](arma::abs(B_ir), gamma, 1)); // composite penalty
    }
    
    out += lambda_i * weight_i * arma::sum(Funs[hfun](sigmas, gamma, 1));
    Sigmas.row(i) = sigmas.t();
  }
  
  return out;
}

// Element-wise conave penalty
double penalty_element(const arma::mat &B,
                       arma::mat &Sigmas,
                       const Rcpp::IntegerVector &d,
					   const Rcpp::NumericVector &lambdas,
                       const std::string &fun_concave,
                       const double &gamma){
  // name of concave function
  std::string hfun = fun_concave; 
  
  double out = {0};
  
  int nDataSets = d.length();
  int R = B.n_cols;
  for(int i = 0; i < nDataSets; ++i){
    
    // index for the ith data set
    arma::uvec indexes = index_Xi(i, d);
    
    // weight for the ith data set
	double lambda_i = lambdas(i);
    int weight_i = 1; // weight when element-wise L1 norm is used
    
    arma::vec sigmas(R);
    for(int r = 0; r < R; ++r){
      const arma::vec &B_ir = B(arma::span(indexes(0), indexes(1)), arma::span(r, r));
      sigmas(r) = arma::sum(arma::abs(B_ir)); // sigma_{lr} = ||b_{lr}||_1
    }
    
    out += lambda_i * weight_i * arma::sum(Funs[hfun](sigmas, gamma, 1));
    Sigmas.row(i) = sigmas.t();
  }
  
  return out;
}

// This is an implementaion of fast verion trace function.
// This function will compute the trace of two matrices.
double trace_fast(const arma::mat &X, const arma::mat &Y) {
  int n = X.n_rows, p = X.n_cols;
  
  // X and Y should have same size
  if ( !(n == Y.n_rows && p == Y.n_cols) )
    throw std::range_error("The two matrices of a trace function are not equal");
  
  if(n > p){
    return arma::trace(X.t() * Y);
  }else{
    return arma::trace(Y * X.t());
  }
}


// variation explained ratios
Rcpp::NumericMatrix varExp_Gaussian(const arma::mat &X,
                                    const Rcpp::IntegerVector &d,
                                    const arma::rowvec &mu,
                                    const arma::mat &A,
                                    const arma::mat &B,
                                    const arma::imat &W){
  // parameter used
  int n = X_i.n_rows;
  int nDataSets = d.length();
  int R = A.n_cols;
  
  // compute the loglikelihood of mle and null model
  arma::mat X_centered = X - arma::ones<arma::vec>(n) * mu;
  arma::mat WX = W % X_centered;
  
  arma::mat E_hat = X_centered - A * B.t();
  arma::mat WE_hat = W % E_hat;
  
  Rcpp::NumericMatrix NvarExp_PCs(nDataSets, R+1);
  
  for (int i = 0; i < nDataSets; ++i){
    arma::uvec indexes = index_Xi(i, d);
    const arma::mat &WX_i = WX.cols(indexes(0), indexes(1));
    const arma::mat &WE_hat_i = WE_hat.cols(indexes(0), indexes(1));
    const arma::mat &W_i = W.cols(indexes(0), indexes(1));
    
    // likelihood of the null model for ith data set
    double l_null = std::pow(arma::norm(WX_i, "fro"), 2); // null model
    
    // likelihood of the full model for ith data set
    double l_model = std::pow(arma::norm(WE_hat_i, "fro"), 2); // full model
    
    // compute the least squares of an individual PC
    Rcpp::NumericVector l_PCs(R, 0);
    
    for(int r = 0; r < R; ++r){
      const arma::vec &Ar = A.col(r);
      arma::vec Bir = B(arma::span(indexes(0), indexes(1)), arma::span(r, r));
      arma::mat WPCr = W_i % (Ar * Bir.t());
      
      l_PCs(r) = l_null - 2 * Bir.t() * WX_i.t() * Ar +  Ar.t() * WPCr * Bir;
    }
    
    // compute variation explained by each PC
    Rcpp::NumericVector varExp_PCs = (1 - l_PCs/l_null) * 100;
    
    // total variation explained
    double varExp_total = (1 - l_model/l_null) * 100;
    varExp_PCs.push_back(varExp_total);
    
    NvarExp_PCs(i, _) = varExp_PCs;
  }
  
  // define the column names
  Rcpp::CharacterVector col_names(R + 1);
  for(int r = 0; r < R; ++r){
    col_names(r) = std::to_string((r + 1)) + " PC";
  }
  col_names(R) = "total";
  Rcpp::colnames(NvarExp_PCs) = col_names;
  
  return NvarExp_PCs;
}


