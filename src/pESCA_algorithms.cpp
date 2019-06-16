#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <math.h>
#include <algorithm>

#include "concave_functions.h"
#include "log_partations.h"
#include "modules_pESCA_algorithm.h"

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]


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
Rcpp::NumericVector varExp_Gaussian(const arma::mat &X,
                                    const arma::rowvec &mu,
                                    const arma::mat &A,
                                    const arma::mat &B,
                                    const arma::imat &W){
  // parameter used
  int n = X.n_rows;
  
  // compute the loglikelihood of mle and null model
  arma::mat X_centered = X - arma::ones<arma::vec>(n) * mu;
  arma::mat WX = W % X_centered;
  
  // likelihood of the null model
  double l_null = std::pow(arma::norm(WX, "fro"), 2); // null model
  
  // likelihood of the full model
  arma::mat E_hat = X_centered - A * B.t();
  arma::mat WE_hat = W % E_hat;
  double l_model = std::pow(arma::norm(WE_hat, "fro"), 2); // full model
  
  // compute the least squares of an individual PC
  int R = A.n_cols;
  Rcpp::NumericVector l_PCs(R, 0);
  
  for(int r = 0; r < R; ++r){
    const arma::vec &Ar = A.col(r);
    const arma::vec &Br = B.col(r);
    arma::mat WPCr = W % (Ar * Br.t());
    
    l_PCs(r) = l_null - 2 * Br.t() * WX.t() * Ar +  Ar.t() * WPCr * Br;
  }
  
  // compute variation explained by each PC
  Rcpp::NumericVector varExp_PCs = (1 - l_PCs/l_null) * 100;
  
  // total variation explained
  double varExp_total = (1 - l_model/l_null) * 100;
  varExp_PCs.push_back(varExp_total);
  
  // define the column names
  Rcpp::CharacterVector vec_names(R + 1);
  for(int r = 0; r < R; ++r){
    vec_names(r) = std::to_string((r + 1)) + " PC";
  }
  vec_names(R) = "total";
  varExp_PCs.names() = vec_names;
  
  return varExp_PCs;
}

// map the names of update_B steps 
// in string format to the corresponding function pointer
std::map<std::string, 
         std::function<void(arma::mat, 
                            arma::mat,
                            arma::mat,
                            arma::mat,
                            Rcpp::IntegerVector,
                            std::string,
                            Rcpp::NumericVector,
                            Rcpp::NumericVector,
                            Rcpp::NumericVector,
                            double)> > Update_Bs = 
                              {{"update_B_L2", update_B_L2},
                               {"update_B_L1", update_B_L1},
                               {"update_B_composite", update_B_composite},
                               {"update_B_elment", update_B_elment}
                              };

// map the names of used penalty function  
// in string format to the corresponding function pointer
std::map<std::string, 
         std::function<double(arma::mat,
                              arma::mat,
                              Rcpp::IntegerVector,
                              Rcpp::NumericVector,
                              std::string,
                              double)> > Penalties = 
                                {{"penalty_L2", penalty_L2},
                                {"penalty_L1", penalty_L1},
                                {"penalty_composite", penalty_composite},
                                {"penalty_element", penalty_element}
                                };
                                                

// map the names of the log partitation functions and it gradient   
// in string format to the corresponding function pointer
std::map<std::string, 
         std::function<arma::mat(arma::mat)> > Log_parts =
                                                 {{"log_part_B", log_part_B},
                                                 {"log_part_B_g", log_part_B_g},
                                                 {"log_part_G", log_part_G},
                                                 {"log_part_G_g", log_part_G_g},
                                                 {"log_part_P", log_part_P},
                                                 {"log_part_P_g", log_part_P_g}
                                                 };

// pESCA models
// [[Rcpp::export]]
Rcpp::List pESCA_C(arma::mat X,
                   const Rcpp::IntegerVector &d,
                   const Rcpp::CharacterVector &dataTypes,
                   const Rcpp::NumericVector &lambdas,
                   const std::string &penalty,
                   const std::string &fun_concave,
                   const Rcpp::List &opts){
  // default parameters
  double tol_obj = {1e-6}, gamma = {1.0};
  int maxit = {1000}, rand_start = {0}, thr_path = {0}, quiet = {1};
  if(opts.containsElementNamed("tol_obj")) tol_obj = opts["tol_obj"];
  if(opts.containsElementNamed("maxit"))    maxit  = opts["maxit"];
  if(opts.containsElementNamed("gamma"))    gamma  = opts["gamma"];
  if(opts.containsElementNamed("rand_start")) rand_start = opts["rand_start"];
  if(opts.containsElementNamed("thr_path")) thr_path = opts["thr_path"];
  if(opts.containsElementNamed("quiet"))    quiet  = opts["quiet"];
  
  // number of data sets, size of each data set
  int nDataSets = d.length(); // number of data sets
  int n = X.n_rows;           // number of samples
  int sumd = X.n_cols;        // total number of variables
  
  // default dispersion parameters alphas and number of PCs
  Rcpp::NumericVector alphas(nDataSets, 1);
  int R = std::round(0.5 * Rcpp::min(n, Rcpp::min(d)));
  if(opts.containsElementNamed("alphas")) alphas = opts["alphas"];
  if(opts.containsElementNamed("R")) R = opts["R"];
  
  // form weighting matrix
  // setting NA to 0
  arma::imat W(n, sumd, arma::fill::zeros); // weighting matrix
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < sumd; ++j){
      if(std::isfinite(X(i,j))){
        W(i,j) = 1;
      }
    }
  }
  X.replace(arma::datum::nan, 0); // remove missing elements
  
  // using svds from RSpectra package
  Rcpp::Environment pkg = Rcpp::Environment::namespace_env("RSpectra");
  Rcpp::Function svds_f = pkg["svds"];
  
  // initialization
  // using random numbers to init parameters
  arma::rowvec mu(sumd, arma::fill::zeros);
  arma::mat A(n, R, arma::fill::randn);
  arma::mat B(sumd, R, arma::fill::randn);
  
  if(opts.containsElementNamed("A0")){ // use imputted initialization
    mu = Rcpp::as<arma::rowvec>(opts["mu0"]);
    A  = Rcpp::as<arma::mat>(opts["A0"]);
    B  = Rcpp::as<arma::mat>(opts["B0"]);
  }else if(rand_start == 0){ // use SCA model as initialization
    if(std::find(dataTypes.begin(), dataTypes.end(), "P") == dataTypes.end())
    { // Possion distribution is not used
      mu = arma::mean(X, 0);
      arma::mat X_centered = X - arma::ones<arma::vec>(n) * mu;
      Rcpp::List svd_tmp = svds_f(X_centered, R);
      A = Rcpp::as<arma::mat>(svd_tmp["u"]);
      arma::vec S = Rcpp::as<arma::vec>(svd_tmp["d"]);
      arma::mat V = Rcpp::as<arma::mat>(svd_tmp["v"]);
      B = V * arma::diagmat(S);
    } else{
      arma::mat X_tmp = X;
      for(int i = 0; i < dataTypes.length(); ++i){
        if(dataTypes(i) == "P"){
          arma::uvec indexes = index_Xi(i, d);
          X_tmp.cols(indexes(0), indexes(1)) = arma::log(X.cols(indexes(0), indexes(1)) + 1);
        }
      }
      mu = arma::mean(X_tmp, 0);
      arma::mat X_centered = X_tmp - arma::ones<arma::vec>(n) * mu;
      Rcpp::List svd_tmp = svds_f(X_centered, R);
      A = Rcpp::as<arma::mat>(svd_tmp["u"]);
      arma::vec S = Rcpp::as<arma::vec>(svd_tmp["d"]);
      arma::mat V = Rcpp::as<arma::mat>(svd_tmp["v"]);
      B = V * arma::diagmat(S);
    }
  }
  
  arma::mat Theta = arma::ones<arma::vec>(n) * mu + A * B.t();
  
  // initial value of loss function
  // specify penalty function
  std::string pfun = "penalty_" + penalty;
  
  double f_obj = {0}, g_obj = {0};
  arma::mat Sigmas(nDataSets, R, arma::fill::zeros);
  for(int i = 0; i < nDataSets; ++i){
    arma::uvec indexes = index_Xi(i, d);
    const arma::mat &X_i = X.cols(indexes(0), indexes(1));
    const arma::imat &W_i = W.cols(indexes(0), indexes(1));
    const arma::mat &Theta_i = Theta.cols(indexes(0), indexes(1));
    double alpha_i = alphas(i);
      
    // loss function for fitting the ith data set
    // specify log-partiton function for the ith data set
    std::string log_part = "log_part_" + dataTypes(i);
    
    f_obj += (1/alpha_i) * (trace_fast(W_i, Log_parts[log_part](Theta_i)) - 
                            trace_fast(Theta_i, X_i));
  }
  
  // penalty for the loading matrix B
  // update Sigmas matrix
  g_obj = Penalties[pfun](B, Sigmas, d, lambdas, fun_concave, gamma);
  
  // record loss and penalty for diagnose purpose
  Rcpp::NumericVector hist_objs, f_objs, g_objs, rel_objs;
  
  // inital values for loss function and penalty
  double rel_obj = {0}, obj0 = {0}, obj = {0};
  obj0 = f_obj + g_obj; // objective + penalty
  f_objs.push_back(f_obj);    // objective
  g_objs.push_back(g_obj);    // penalty
  hist_objs.push_back(obj0);
  
  // iterations
  int k = {0};
  while(k <= maxit){
    if(quiet == 0) Rprintf("%i th iteration \n", k);
    
    // majorizaiton step for p_ESCA model
    //--- form Hk ---
    //--- update mu ---
    //--- form JHk ---
    arma::mat JHk(n, sumd, arma::fill::zeros);
    Rcpp::NumericVector rhos(nDataSets); // form rhos, the Lipshize constant for each data types
    arma::vec cs(sumd, arma::fill::ones); // scaling factors
    
    for(int i = 0; i < nDataSets; ++i){
      arma::uvec indexes = index_Xi(i, d);
      const arma::mat &X_i = X.cols(indexes(0), indexes(1));
      const arma::imat &W_i = W.cols(indexes(0), indexes(1));
      const arma::mat &Theta_i = Theta.cols(indexes(0), indexes(1));
      double alpha_i = alphas(i);
      double lambda_i = lambdas(i);
      
      // specify the gradient of the log-partiton function
      std::string log_part_g = "log_part_" + dataTypes(i) + "_g";
      
      // form rhos, the Lipshize constant for each data types
      double rho_i = {1.0};
      if(dataTypes(i) == "B"){
        rho_i = 0.25;
      }else if(dataTypes(i) == "P"){
        rho_i = arma::exp(Theta_i).max();
      }
      rhos(i) = rho_i;
      
      // form Hk_i
      arma::mat Hk_i = Theta_i - (1/rho_i) * (W_i % (Log_parts[log_part_g](Theta_i) - X_i));
      
      // update mu_i
      arma::rowvec mu_i = arma::mean(Hk_i, 0);
      mu.cols(indexes(0), indexes(1)) = mu_i;
      
      // form JHk_i
      JHk.cols(indexes(0), indexes(1)) = Hk_i - arma::ones<arma::vec>(n) * mu_i;
      
      // form scaling factors for scaled_JHk_i, scaled_Bk_i
      cs.cols(indexes(0), indexes(1)).fill(std::sqrt(rho_i/alpha_i));
    }
    
    // update A
    arma::mat A_tmp = JHk * diagmat(arma::pow(cs, 2)) * B;
    arma::mat U, V;
    arma::vec s;
    svd_econ(U, s, V, A_tmp);
    A = U * V.t();
    
    // update B
    std::string up_B_fun = "update_B_" + penalty;
    Update_Bs[up_B_fun](JHk, A, B, Sigmas, d, fun_concave, alphas, rhos, lambdas, gamma);
    
    // diagnostics
    Theta = arma::ones<arma::vec>(n) * mu + A * B.t();
    
    f_obj = {0};
    g_obj = {0};
    for(int i = 0; i < nDataSets; ++i){
      arma::uvec indexes = index_Xi(i,d);
      const arma::mat &X_i = X.cols(indexes(0), indexes(1));
      const arma::imat &W_i = W.cols(indexes(0), indexes(1));
      const arma::mat &Theta_i = Theta.cols(indexes(0), indexes(1));
      double alpha_i = alphas(i);
      
      // loss function for fitting ith data set
      // specify log-partiton function for the ith data set
      std::string log_part = "log_part_" + dataTypes(i);
      
      f_obj += (1/alpha_i) * (trace_fast(W_i, Log_parts[log_part](Theta_i)) - 
                              trace_fast(Theta_i, X_i));
    }
    
    // penalty for the loading matrix B
    g_obj = Penalties[pfun](B, Sigmas, d, lambdas, fun_concave, gamma);
    
    // new values for loss function and penalty
    obj = f_obj + g_obj; // objective + penalty
    f_objs.push_back(f_obj);    // objective
    g_objs.push_back(g_obj);    // penalty
    hist_objs.push_back(obj);
    
    // reporting
    rel_obj = (obj0-obj)/abs(obj0); // relative change of loss function
    rel_objs.push_back(rel_obj);
    
    // stopping checks
    if(k != 0 && rel_obj < tol_obj) break;
    
    // remove the all zeros columns to simplify the computation and save memory
    if(thr_path == 0){
      auto nonZeros_index = arma::mean(Sigmas,0) > 0;
      if(arma::sum(nonZeros_index) > 3){
        A = A.cols(nonZeros_index);
        B = B.cols(nonZeros_index);
        Sigmas = Sigmas.cols(nonZeros_index);
        R = B.n_cols;
      }
    }
    
    // save previous results
    obj0 = obj;
    ++k;
  }
  
  // output
  Rcpp::List diagnose = Rcpp::List::create(Rcpp::Named("hist_objs") = hist_objs,
                                           Rcpp::Named("f_objs")    = f_objs,
                                           Rcpp::Named("g_objs")    = g_objs,
                                           Rcpp::Named("rel_objs")  = rel_objs);
  Rcpp::List result = Rcpp::List::create(Rcpp::Named("mu") = mu,
                                         Rcpp::Named("A") = A,
                                         Rcpp::Named("B") = B,
                                         Rcpp::Named("Sigmas") = Sigmas,
                                         Rcpp::Named("iter") = k,
                                         Rcpp::Named("diagnose") = diagnose);
  return result;
  
}







