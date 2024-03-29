#include <iostream>
#include <string>
#include <map>
#include <math.h>
#include <algorithm>

#include "concave_functions.h"
#include "log_partations.h"
#include "modules_pESCA_algorithm.h"

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]

// map the names of update_B steps 
// in string format to the corresponding function pointer
typedef void (*update_B_type)(const arma::mat &,
              const arma::mat &,
              arma::mat &,
              const arma::mat &,
              const Rcpp::IntegerVector &,
              const std::string &,
              const Rcpp::NumericVector &,
              const Rcpp::NumericVector &,
              const Rcpp::NumericVector &,
              const double &);

std::map<std::string, update_B_type> Update_Bs = 
                              {{"update_B_L2", update_B_L2},
                              {"update_B_L1", update_B_L1},
                              {"update_B_composite", update_B_composite},
                              {"update_B_element", update_B_element}
                              };

// map the names of used penalty function  
// in string format to the corresponding function pointer
typedef double(*penalty_type)(const arma::mat &,
                             arma::mat &,
                             const Rcpp::IntegerVector &,
                             const Rcpp::NumericVector &,
                             const std::string &,
                             const double &);

std::map<std::string, penalty_type> Penalties = 
                                {{"penalty_L2", penalty_L2},
                                {"penalty_L1", penalty_L1},
                                {"penalty_composite", penalty_composite},
                                {"penalty_element", penalty_element}
                                };
                                                
// map the names of the log partitation functions and it gradient   
// in string format to the corresponding function pointer
typedef arma::mat (*log_part_type)(const arma::mat &);

std::map<std::string, log_part_type > Log_parts =
                                                 {{"log_part_B", log_part_B},
                                                 {"log_part_B_g", log_part_B_g},
                                                 {"log_part_G", log_part_G},
                                                 {"log_part_G_g", log_part_G_g},
                                                 {"log_part_P", log_part_P},
                                                 {"log_part_P_g", log_part_P_g}
                                                 };
 
//' C++ implementation of penalized exponential family simultaneous component analysis (pESCA) model
//'
//' This is the main function for construncting a pESCA model on multiple data
//' sets. The potential different data types in these data sets are tackled by
//' the assumption of exponential family distribution. Gaussian for quantitative
//' data, Bernoulli for binary data and Poisson for count data. Although the option
//' for count data using Poisson distribution is included in the algorithm, we recommend
//' to do variance stabilizing transformation on the count data, such as Next-Gen
//' sequencing data, and then use the transformed data as quantitative data sets. The
//' details of the developed algorithm can be found in \url{https://arxiv.org/abs/1902.06241}.
//' 
//' @inheritParams pESCA
//' 
//' @param X the matrix to hold all the data sets
//' @param d the vector to indicate the number of variables in each data set
//'
//' @import RSpectra
//'
//' @examples
//' \dontrun{
//' # Suppose we have three data sets X1, X2, X3
//' # They are quantitative, quantitative and binary matrices
//' pESCA_C(X = X, d = d,
//'               dataTypes = c("G", "G", "B"),
//'               lambdas = c(20, 20, 10),
//'               penalty = 'L2',
//'               fun_concave = 'gdp',
//'               opts = list())
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List pESCA_C(arma::mat &X,
                   const Rcpp::IntegerVector &d,
                   const Rcpp::CharacterVector &dataTypes,
                   const Rcpp::NumericVector &lambdas,
                   const std::string &penalty,
                   const std::string &fun_concave,
                   const Rcpp::List &opts){
  // default parameters
  double tol_obj = {1e-6}, gamma = {1.0};
  int maxit = {1000}, rand_start = {0}, thr_path = {0}, quiet = {1}, fit_mode = {1};
  if(opts.containsElementNamed("tol_obj")) tol_obj = opts["tol_obj"];
  if(opts.containsElementNamed("maxit"))    maxit  = opts["maxit"];
  if(opts.containsElementNamed("gamma"))    gamma  = opts["gamma"];
  if(opts.containsElementNamed("rand_start")) rand_start = opts["rand_start"];
  if(opts.containsElementNamed("thr_path")) thr_path = opts["thr_path"];
  if(opts.containsElementNamed("quiet"))    quiet = opts["quiet"];
  if(opts.containsElementNamed("fit_mode")) fit_mode  = opts["fit_mode"];
  
  // number of data sets, size of each data set
  int nDataSets = d.length(); // number of data sets
  int n = X.n_rows;           // number of samples
  int sumd = X.n_cols;        // total number of variables
  
  // default dispersion parameters alphas and number of PCs
  Rcpp::NumericVector alphas(nDataSets);
  alphas.fill(1);
  if(opts.containsElementNamed("alphas")) alphas = opts["alphas"];
  
  int d_min = Rcpp::min(d);
  int R = std::round(0.5 * ((n < d_min) ? n : d_min));
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
  double f_obj = {0};
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
  // specify penalty function
  std::string pfun = "penalty_" + penalty;
  
  double g_obj = {0};
  arma::mat Sigmas(nDataSets, R, arma::fill::zeros);
  
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
      auto nonZeros_index = arma::mean(Sigmas, 0) > 0;
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
  if(fit_mode == 1){
    Rcpp::NumericMatrix NvarExp_PCs = varExp_Gaussian(X, d, mu, A, B, W);
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("mu") = mu,
                                           Rcpp::Named("A") = A,
                                           Rcpp::Named("B") = B,
                                           Rcpp::Named("Sigmas") = Sigmas,
                                           Rcpp::Named("iter") = k,
                                           Rcpp::Named("diagnose") = diagnose,
                                           Rcpp::Named("varExp_PCs") = NvarExp_PCs);
    return result;
    
  }else{
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("mu") = mu,
                                           Rcpp::Named("A") = A,
                                           Rcpp::Named("B") = B,
                                           Rcpp::Named("Sigmas") = Sigmas,
                                           Rcpp::Named("iter") = k,
                                           Rcpp::Named("diagnose") = diagnose);
    return result;
  }
}



