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
std::map<std::string, std::function<arma::vec(arma::vec,double,double)> > concave_funs = 
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
  if(i==0){
    indexes(0) = 0;
    indexes(1) = (ds[i]-1);
  }else{
    indexes(0) = Rcpp::sum(ds[Rcpp::seq(0,i-1)]);
    indexes(1) = Rcpp::sum(ds[Rcpp::seq(0,i)])-1;
  }
  
  return indexes;
}

// Updating loading matrix B when conave L2 norm penalty is used
arma::mat update_B_L2(const arma::mat &JHk,
                      const arma::mat &A,
                      const arma::mat &B0,
                      const arma::mat &Sigmas0,
                      const Rcpp::IntegerVector &d,
                      std::string fun_name,
                      const Rcpp::NumericVector &alphas,
                      const Rcpp::NumericVector &rhos,
                      const Rcpp::NumericVector &lambdas,
                      double gamma){
  int sumd = Rcpp::sum(d);
  int nDataSets = d.length();
  int n = A.n_rows;
  int R = A.n_cols;
  fun_name += "_sg";
  
  arma::mat B(size(B0),arma::fill::zeros);
  for(int i=0; i<nDataSets; ++i){
    arma::uvec indexes(2);
    indexes = index_Xi(i,d);
    const arma::mat &JHk_i = JHk.cols(indexes(0),indexes(1));
    arma::mat JHkitA = JHk_i.t() * A;
    double alpha_i = alphas[i];
    double rho_i = rhos[i];
    double weight_i = std::sqrt(d[i]); // weight for L2 norm 
    double lambda_i = lambdas[i] * weight_i * alpha_i / rho_i;
    
    // form weights of the penalty according to previous sigma0_ir
    arma::vec sigma0_i = Sigmas0.row(i);
    arma::vec omega_i  = concave_funs[fun_name](sigma0_i,gamma,1); // weights
    
    for(int r=0; r<R; ++r){
      // proximal operator of L2 norm
      const arma::vec &JHkitA_r = JHkitA.col(r);
      double lambda_ir = lambda_i * omega_i(r);
      double JHkitA_r_norm = arma::norm(JHkitA_r, 2);
      
      arma::vec B_ir = (1-(lambda_ir/JHkitA_r_norm)) * JHkitA_r;
      B_ir.elem(find(B_ir<0)).zeros();
      B(arma::span(indexes(0),indexes(1)),arma::span(r,r)) = B_ir;
    }
  }
  
  return B;
}

// Group-wise conave L2 norm penalty
Rcpp::List concave_L2(const arma::mat &B_i,
                      const std::string &fun_name,
                      double gamma){
  // the number of PCs
  int R = B_i.n_cols;
  
  // weight for the ith data set
  double weight_i = std::sqrt(B_i.n_rows); // weight when L2 norm is used
  
  double out{0};
  arma::vec sigmas(R);
  for(int r=0; r<R; ++r){
    const arma::vec &B_ir = B_i.col(r);
    sigmas(r) = arma::norm(B_ir,2); // sigma_{lr} = ||b_{lr}||_2
  }
  
  out += weight_i * arma::sum(concave_funs[fun_name](sigmas,gamma,1));
  
  Rcpp::List result = Rcpp::List::create(Rcpp::Named("out") = out,
                                         Rcpp::Named("sigmas") = sigmas.t());
  return result;
}

// Updating loading matrix B with conave L1 norm penalty
arma::mat update_B_L1(const arma::mat &JHk,
                      const arma::mat &A,
                      const arma::mat &B0,
                      const arma::mat &Sigmas0,
                      const Rcpp::IntegerVector &d,
                      std::string fun_name,
                      const Rcpp::NumericVector &alphas,
                      const Rcpp::NumericVector &rhos,
                      const Rcpp::NumericVector &lambdas,
                      double gamma){
  int sumd = Rcpp::sum(d);
  int nDataSets = d.length();
  int n = A.n_rows;
  int R = A.n_cols;
  fun_name += "_sg";
  
  arma::mat B(size(B0),arma::fill::zeros);
  for(int i=0; i<nDataSets; ++i){
    arma::uvec indexes(2);
    indexes = index_Xi(i,d);
    const arma::mat &JHk_i = JHk.cols(indexes(0),indexes(1));
    arma::mat JHkitA = JHk_i.t() * A;
    double alpha_i = alphas[i];
    double rho_i = rhos[i];
    double weight_i = d[i]; // weight for L1 norm 
    double lambda_i = lambdas[i] * weight_i * alpha_i / rho_i;
    
    // form weights of the penalty according to previous sigma0_ir
    arma::vec sigma0_i = Sigmas0.row(i);
    arma::vec omega_i  = concave_funs[fun_name](sigma0_i,gamma,1); // weights
    
    for(int r=0; r<R; ++r){
      // proximal operator of L1 norm
      const arma::vec &JHkitA_r = JHkitA.col(r);
      double lambda_ir = lambda_i * omega_i(r);
      
      arma::vec B_ir_pre = arma::abs(JHkitA_r) - lambda_ir;
      B_ir_pre.elem(find(B_ir_pre<0)).zeros();
      arma::vec B_ir = arma::sign(JHkitA_r) * B_ir_pre;
      B(arma::span(indexes(0),indexes(1)),arma::span(r,r)) = B_ir;
    }
  }
  
  return B;
}

// Group-wise conave L1 norm penalty
Rcpp::List concave_L1(const arma::mat &B_i,
                      const std::string &fun_name,
                      double gamma){
  // the number of PCs
  int R = B_i.n_cols;
  
  // weight for ith data set
  int weight_i = B_i.n_rows; // weight when L1 norm is used
  
  double out{0};
  arma::vec sigmas(R);
  for(int r=0; r<R; ++r){
    const arma::vec &B_ir = B_i.col(r);
    sigmas(r) = arma::sum(arma::abs(B_ir)); // sigma_{lr} = ||b_{lr}||_1
  }
  
  out += weight_i * arma::sum(concave_funs[fun_name](sigmas,gamma,1));
  
  Rcpp::List result = Rcpp::List::create(Rcpp::Named("out") = out,
                                         Rcpp::Named("sigmas") = sigmas.t());
  
  return result;
}

// Updating loading matrix B with the composite concave penalty
arma::mat update_B_composite(const arma::mat &JHk,
                      const arma::mat &A,
                      const arma::mat &B0,
                      const arma::mat &Sigmas0,
                      const Rcpp::IntegerVector &d,
                      std::string fun_name,
                      const Rcpp::NumericVector &alphas,
                      const Rcpp::NumericVector &rhos,
                      const Rcpp::NumericVector &lambdas,
                      double gamma){
  int sumd = Rcpp::sum(d);
  int nDataSets = d.length();
  int n = A.n_rows;
  int R = A.n_cols;
  fun_name += "_sg";
  
  arma::mat B(size(B0),arma::fill::zeros);
  for(int i=0; i<nDataSets; ++i){
    arma::uvec indexes(2);
    indexes = index_Xi(i,d);
    const arma::mat &JHk_i = JHk.cols(indexes(0),indexes(1));
    arma::mat JHkitA = JHk_i.t() * A;
    double alpha_i = alphas[i];
    double rho_i = rhos[i];
    double weight_i = d[i]; // weight for L1 norm 
    double lambda_i = lambdas[i] * weight_i * alpha_i / rho_i;
    
    // form weights of the penalty according to previous sigma0_ir
    // weights of the outer layer
    arma::vec sigma0_i = Sigmas0.row(i);
    arma::vec omega_i_outer = concave_funs[fun_name](sigma0_i,gamma,1); 
    
    for(int r=0; r<R; ++r){
      // weights of the inner layer
      arma::vec sigma0_ir_vec = arma::abs(B0(arma::span(indexes(0),indexes(1)),arma::span(r,r)));
      arma::vec omega_ir_inner = concave_funs[fun_name](sigma0_ir_vec,gamma,1);
      
      // total weights
      arma::vec omega_ir_vec = omega_i_outer(r) * omega_ir_inner;
      
      // proximal operator of L1 norm
      const arma::vec &JHkitA_r = JHkitA.col(r);
      arma::vec lambda_ir_vec = lambda_i * omega_ir_vec;
      
      arma::vec B_ir_pre = arma::abs(JHkitA_r) - lambda_ir_vec;
      B_ir_pre.elem(find(B_ir_pre<0)).zeros();
      arma::vec B_ir = arma::sign(JHkitA_r) * B_ir_pre;
      B(arma::span(indexes(0),indexes(1)),arma::span(r,r)) = B_ir;
    }
  }
  
  return B;
}

// Composition of group-wise and element-wise conave penalty
Rcpp::List concave_composite(const arma::mat &B_i,
                             const std::string &fun_name,
                             double gamma){
  // the number of PCs
  int R = B_i.n_cols;
  
  // weight for ith data set
  int weight_i = B_i.n_rows; // weight when composite L1 norm is used
  
  double out = 0;
  arma::vec sigmas(R);
  for(int r=0; r<R; ++r){
    const arma::vec &B_ir = B_i.col(r);
    sigmas(r) = arma::sum(concave_funs[fun_name](arma::abs(B_ir),gamma,1));
  }
  
  out += weight_i * arma::sum(concave_funs[fun_name](sigmas,gamma,1));
  
  Rcpp::List result = Rcpp::List::create(Rcpp::Named("out") = out,
                                         Rcpp::Named("sigmas") = sigmas.t());
  
  return result;
}


// Updating loading matrix B with the element-wise concave penalty
arma::mat update_B_elment(const arma::mat &JHk,
                          const arma::mat &A,
                          const arma::mat &B0,
                          const arma::mat &Sigmas0,
                          const Rcpp::IntegerVector &d,
                          std::string fun_name,
                          const Rcpp::NumericVector &alphas,
                          const Rcpp::NumericVector &rhos,
                          const Rcpp::NumericVector &lambdas,
                          double gamma){
  int sumd = Rcpp::sum(d);
  int nDataSets = d.length();
  int n = A.n_rows;
  int R = A.n_cols;
  fun_name += "_sg";
  
  arma::mat B(size(B0),arma::fill::zeros);
  for(int i=0; i<nDataSets; ++i){
    arma::uvec indexes(2);
    indexes = index_Xi(i,d);
    const arma::mat &JHk_i = JHk.cols(indexes(0),indexes(1));
    arma::mat JHkitA = JHk_i.t() * A;
    double alpha_i = alphas[i];
    double rho_i = rhos[i];
    double weight_i = 1; // weight element-wise penalty 
    double lambda_i = lambdas[i] * weight_i * alpha_i / rho_i;
    
    for(int r=0; r<R; ++r){
      // form weights of the penalty according to previous sigma0_ir
      arma::vec sigma0_ir_vec = arma::abs(B0(arma::span(indexes(0),indexes(1)),arma::span(r,r)));
      arma::vec omega_ir_vec  = concave_funs[fun_name](sigma0_ir_vec,gamma,1);
      
      // proximal operator of L1 norm
      const arma::vec &JHkitA_r = JHkitA.col(r);
      arma::vec lambda_ir_vec = lambda_i * omega_ir_vec;
      
      arma::vec B_ir_pre = arma::abs(JHkitA_r) - lambda_ir_vec;
      B_ir_pre.elem(find(B_ir_pre<0)).zeros();
      arma::vec B_ir = arma::sign(JHkitA_r) * B_ir_pre;
      B(arma::span(indexes(0),indexes(1)),arma::span(r,r)) = B_ir;
    }
  }
  
  return B;
}

// Element-wise conave penalty
Rcpp::List concave_element(const arma::mat &B_i,
                           const std::string &fun_name,
                           double gamma){
  // the number of PCs
  int R = B_i.n_cols;
  
  // weight for ith data set
  int weight_i = 1; // weight when element-wise L1 norm is used
  
  double out = 0;
  arma::vec sigmas(R);
  for(int r=0; r<R; ++r){
    const arma::vec &B_ir = B_i.col(r);
    sigmas(r) = arma::sum(arma::abs(B_ir));
  }
  
  out += weight_i * arma::sum(concave_funs[fun_name](sigmas,gamma,1));
  
  Rcpp::List result = Rcpp::List::create(Rcpp::Named("out") = out,
                                         Rcpp::Named("sigmas") = sigmas.t());
  
  return result;
}