#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <math.h>
#include <algorithm>
#include "concave_penalties.h"
#include "log_partations.h"
#include "modules_pESCA_algorithm.h"

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[RcPP::plugins("cpp11")]]

// map the names of update_B steps 
// in string format to the corresponding function pointer
std::map<std::string, std::function<arma::mat(arma::mat,
                                              arma::mat,
                                              arma::mat,
                                              arma::mat,
                                              Rcpp::IntegerVector,
                                              std::string,
                                              Rcpp::NumericVector,
                                              Rcpp::NumericVector,
                                              Rcpp::NumericVector,
                                              double)> > update_B =
                                                {{"update_B_L2", update_B_L2},
                                                {"update_B_L1", update_B_L1},
                                                {"update_B_composite", update_B_composite},
                                                {"update_B_elment", update_B_elment}
                                                };

// map the names of used penalty function  
// in string format to the corresponding function pointer
std::map<std::string, std::function<Rcpp::List(arma::mat,
                                               std::string,
                                               double)> > penalty_concave =
                                                {{"concave_L2", concave_L2},
                                                {"concave_L1", concave_L1},
                                                {"concave_composite", concave_composite},
                                                {"concave_elment", concave_element}
                                                };

// map the names of the log partitation functions and it gradient   
// in string format to the corresponding function pointer
std::map<std::string, std::function<arma::mat(arma::mat)> > log_part =
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
                   Rcpp::IntegerVector d,
                   Rcpp::CharacterVector dataTypes,
                   Rcpp::NumericVector lambdas,
                   std::string penalty,
                   std::string fun_name,
                   Rcpp::List opts){
  // extract used parameters
  double tol_obj = opts["tol_obj"];
  int      maxit = opts["maxit"];
  double   gamma = opts["gamma"];
  int      quiet = opts["quiet"];
  int rand_start = opts["rand_start"];
  int   thr_path = opts["thr_path"]; 
  int          R = opts["R"];
  Rcpp::NumericVector alphas = opts["alphas"];
  
  // number of data sets, size of each data set
  int nDataSets = d.length(); // number of data sets
  int n = X.n_rows;           // number of samples
  int sumd = X.n_cols;        // total number of variables
  
  // form weighting matrix
  // setting NA to 0
  arma::imat W(n,sumd,arma::fill::zeros); // weighting matrix
  for(int i=0; i<n; ++i){
    for(int j=0; j<sumd; ++j){
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
  // initial parameters
  arma::rowvec mu(sumd, arma::fill::zeros);
  arma::mat A(n, R, arma::fill::randn);
  arma::mat B(sumd, R, arma::fill::randn);
  
  if(opts.containsElementNamed("A0")){ // use imputted initialization
    mu = Rcpp::as<arma::rowvec>(opts["mu0"]);
    A  = Rcpp::as<arma::mat>(opts["A0"]);
    B  = Rcpp::as<arma::mat>(opts["B0"]);
  }else if(rand_start==0){
    if(std::find(dataTypes.begin(),dataTypes.end(), "P") == dataTypes.end())
    { // Possition distribution is not used
      mu = arma::mean(X,0);
      arma::mat X_centered = X - arma::ones<arma::vec>(n) * mu;
      Rcpp::List svd_tmp = svds_f(X_centered,R);
      A = Rcpp::as<arma::mat>(svd_tmp["u"]);
      arma::vec S = Rcpp::as<arma::vec>(svd_tmp["d"]);
      arma::mat V = Rcpp::as<arma::mat>(svd_tmp["v"]);
      B = V * arma::diagmat(S);
    } else{
      arma::mat X_tmp = X;
      for(int i=0; i<dataTypes.length(); ++i){
        if(dataTypes(i)=="P"){
          arma::uvec indexes = index_Xi(i,d);
          X_tmp.cols(indexes(0),indexes(1)) = arma::log(X.cols(indexes(0),indexes(1))+1);
        }
      }
      mu = arma::mean(X_tmp,0);
      arma::mat X_centered = X_tmp - arma::ones<arma::vec>(n) * mu;
      Rcpp::List svd_tmp = svds_f(X_centered,R);
      A = Rcpp::as<arma::mat>(svd_tmp["u"]);
      arma::vec S = Rcpp::as<arma::vec>(svd_tmp["d"]);
      arma::mat V = Rcpp::as<arma::mat>(svd_tmp["v"]);
      B = V * arma::diagmat(S);
    }
  }
  
  arma::mat Theta = arma::ones<arma::vec>(n) * mu + A * B.t();
  
  // initial value of loss function
  // specify penalty function
  std::string penalty_fun = "concave_" + penalty;
  std::string update_B_fun = "update_B_" + penalty;
  
  double f_obj0 = 0, g_obj0 = 0;
  arma::mat Sigmas(nDataSets, R, arma::fill::zeros);
  for(int i=0; i<nDataSets; ++i){
    arma::uvec indexes(2);
    indexes = index_Xi(i,d);
    
    const arma::mat &X_i = X.cols(indexes(0),indexes(1));
    const arma::imat &W_i = W.cols(indexes(0),indexes(1));
    const arma::mat &Theta_i = Theta.cols(indexes(0),indexes(1));
    double alpha_i = alphas(i);
    double lambda_i = lambdas(i);
    
    // loss function for fitting ith data set
    // specify log-partiton function for ith data set
    std::string log_part_name = "log_part_" + dataTypes(i);
    f_obj0 += (1/alpha_i) * (arma::trace(W_i * log_part[log_part_name](Theta_i)) -
                             arma::trace(Theta_i * X_i));
    
    // penalty for the ith loading matrix B_l
    const arma::mat &B0_i = B.rows(indexes(0),indexes(1));
    Rcpp::List g_penalty = penalty_concave[penalty_fun](B0_i, fun_name, gamma);
    g_obj0 += lambda_i * Rcpp::as<double>(g_penalty["out"]);
    Sigmas.row(i) = Rcpp::as<arma::rowvec>(g_penalty["sigmas"]);
  }
  
  // record loss and penalty for diagnose purpose
  Rcpp::NumericVector hist_objs, f_objs, g_objs, rel_objs, rel_Thetas;
  
  // inital values for loss function and penalty
  double obj0 = f_obj0 + g_obj0;
  f_objs.push_back(f_obj0);
  g_objs.push_back(g_obj0);
  hist_objs.push_back(obj0);
  
  // iterations
  int k = 0;
  while(k <= maxit){
    if(quiet==0) Rprintf("%i th iteration \n",k);
    
    // majorizaiton step for p_ESCA model
    //--- form Hk ---
    //--- update mu ---
    //--- form JHk ---
    arma::mat JHk(n, sumd, arma::fill::zeros);
    Rcpp::NumericVector rhos(nDataSets);
    arma::vec cs(sumd, arma::fill::ones);
    
    for(int i=0; i<nDataSets; ++i){
      arma::uvec indexes(2);
      indexes = index_Xi(i,d);
      
      const arma::mat &X_i = X.cols(indexes(0),indexes(1));
      const arma::imat &W_i = W.cols(indexes(0),indexes(1));
      const arma::mat &Theta_i = Theta.cols(indexes(0),indexes(1));
      double alpha_i = alphas(i);
      double lambda_i = lambdas(i);
      
      // specify the gradient of the log-partiton function
      std::string log_part_g_name = "log_part_" + dataTypes(i) + "_g";
      
      // form rhos, the Lipshize constant for each data types
      double rho_i{1.0};
      if(dataTypes(i) == "B"){
        rho_i = 0.25;
      }else if(dataTypes(i) == "P"){
        rho_i = arma::exp(Theta_i).max();
      }
      rhos(i) = rho_i;
      
      // form Hk_i
      arma::mat Hk_i = Theta_i - (1/rho_i)*(W_i * (log_part[log_part_g_name](Theta_i) - X_i));
      
      // update mu_i
      arma::rowvec mu_i = arma::mean(Hk_i,0);
      mu.cols(indexes(0),indexes(1)) = mu_i;
      
      // form JHk_i
      JHk.cols(indexes(0),indexes(1)) = Hk_i - arma::ones<arma::vec>(n) * mu_i;
      
      // form scaling factors for scaled_JHk_i, scaled_Bk_i
      cs.cols(indexes(0),indexes(1)).fill(std::sqrt(rho_i/alpha_i));
    }
    
    // update A
    arma::mat A_tmp = JHk * diagmat(arma::pow(cs,2)) * B;
    arma::mat U, V;
    arma::vec s;
    svd_econ(U,s,V,A_tmp);
    A = U * V.t();
    
    // update B
    B = update_B[update_B_fun](JHk, A,B,Sigmas,d,fun_name,alphas,rhos,lambdas,gamma);
    
    // diagnostics
    Theta = arma::ones<arma::vec>(n) * mu + A * B.t();
    
    double f_obj = 0, g_obj = 0;
    //arma::mat Sigmas(nDataSets, R, arma::fill::zeros);
    for(int i=0; i<nDataSets; ++i){
      arma::uvec indexes(2);
      indexes = index_Xi(i,d);
      
      const arma::mat &X_i = X.cols(indexes(0),indexes(1));
      const arma::imat &W_i = W.cols(indexes(0),indexes(1));
      const arma::mat &Theta_i = Theta.cols(indexes(0),indexes(1));
      double alpha_i = alphas(i);
      double lambda_i = lambdas(i);
      
      // loss function for fitting ith data set
      // specify log-partiton function for ith data set
      std::string log_part_name = "log_part_" + dataTypes(i);
      f_obj += (1/alpha_i) * (arma::trace(W_i * log_part[log_part_name](Theta_i)) -
        arma::trace(Theta_i * X_i));
      
      // penalty for the ith loading matrix B_l
      const arma::mat &B0_i = B.rows(indexes(0),indexes(1));
      Rcpp::List g_penalty = penalty_concave[penalty_fun](B0_i, fun_name, gamma);
      g_obj += lambda_i * Rcpp::as<double>(g_penalty["out"]);
      Sigmas.row(i) = Rcpp::as<arma::rowvec>(g_penalty["sigmas"]);
    }
    double obj = f_obj + g_obj; // objective + penalty
    f_objs.push_back(f_obj);  // objective
    g_objs.push_back(g_obj);  // penalty
    hist_objs.push_back(obj); // objective + penalty
    
    double rel_obj = (obj0-obj)/abs(obj0); // relative change of loss function
    
    // remove the all zeros columns to simplify the computation and save memory
    if(thr_path == 0){
      auto nonZeros_index = arma::mean(Sigmas,0) > 0;
      if(arma::sum(nonZeros_index)>3){
        A = A.cols(nonZeros_index);
        B = B.cols(nonZeros_index);
        Sigmas = Sigmas.cols(nonZeros_index);
        R = B.n_cols;
      }
    }
    
    // stopping checks
    if(k != 0 && rel_obj < tol_obj) break;
    
    // save previous results
    obj0 = obj;
    ++k;
  }
  
  // output
  Rcpp::List diagnose = Rcpp::List::create(Rcpp::Named("hist_objs") = hist_objs,
                                           Rcpp::Named("f_objs")    = f_objs,
                                           Rcpp::Named("g_objs")    = g_objs,
                                           Rcpp::Named("rel_objs")  = rel_objs);
  Rcpp::List result = Rcpp::List::create(Rcpp::Named("mu") = mu.t(),
                                         Rcpp::Named("A") = A,
                                         Rcpp::Named("B") = B,
                                         Rcpp::Named("Sigmas") = Sigmas,
                                         Rcpp::Named("iter") = k,
                                         Rcpp::Named("diagnose") = diagnose);
  return result;
  
}







