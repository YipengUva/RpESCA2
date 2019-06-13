#include <iostream>
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[RcPP::plugins("cpp11")]]

//' A svd algorithm with the option for missing values.
//'
//' This function implemented a MM algorithm to fit a svd on a quantitaive data
//' set with missing values. The details of this function can be found
//' in \url{https://arxiv.org/abs/1902.06241}.
//'
//' @param X a quantitative data set
//' @param R the number of PCs
//' @param opts a list contains the setting for the algorithm. \itemize{
//' \item tol_obj: tolerance for relative change of hist_obj, default:1E-6;
//' \item maxit: max number of iterations, default: 1000;
//' }
//'
//' @return This function returns a list contains \itemize{
//' \item U: the left singular vectors
//' \item S: a column vector contains the singular values
//' \item V: the right singular vectors
//' \item iter: the number of iterations used
//' \item diagnose: records hist_obj and rel_obj
//' }
//'
//' @import RSpectra
//' 
//' @examples
//' \dontrun{svd_mis(X,R=3,opts=list())}
//'
//' @export
// [[Rcpp::export]]
Rcpp::List svd_mis(arma::mat X,
                    int R, Rcpp::List opts){
  // default parameters
  double tol_obj{0};
  int maxit{0}, quiet{0};
  if(opts.containsElementNamed("tol_obj")) tol_obj = opts["tol_obj"];
  else tol_obj = 1e-6;
  if(opts.containsElementNamed("maxit"))    maxit  = opts["maxit"];
  else maxit   = 1000;
  if(opts.containsElementNamed("quiet"))    quiet  = opts["quiet"];
  else quiet   = 1;
  
  // form weighting matrix
  int nrows = X.n_rows;
  int ncols = X.n_cols;
  arma::imat W(nrows,ncols,arma::fill::zeros); // weighting matrix
  for(int i=0; i<nrows; ++i){
    for(int j=0; j<ncols; ++j){
      if(std::isfinite(X(i,j))){
        W(i,j) = 1;
      }
    }
  }
  arma::imat W_c = 1-W;
  X.replace(arma::datum::nan, 0); // remove missing elements
  
  // using svds from RSpectra package
  Rcpp::Environment pkg = Rcpp::Environment::namespace_env("RSpectra");
  Rcpp::Function svds_f = pkg["svds"];
  
  // initialization
  // initial parameters
  arma::mat U(nrows,R,arma::fill::zeros);
  arma::vec S(R, arma::fill::ones); 
  arma::mat V(ncols,R,arma::fill::zeros);
  arma::mat Z(size(X));
  
  if(opts.containsElementNamed("U0")){
    U = Rcpp::as<arma::mat>(opts["U0"]);
    S = Rcpp::as<arma::vec>(opts["S0"]);
    V = Rcpp::as<arma::mat>(opts["V0"]);
  }else{
    Rcpp::List svd_tmp = svds_f(X,R);
    U = Rcpp::as<arma::mat>(svd_tmp["u"]);
    S = Rcpp::as<arma::vec>(svd_tmp["d"]);
    V = Rcpp::as<arma::mat>(svd_tmp["v"]);
  }
  Z = U * diagmat(S) * V.t();
  
  // specify structure to hold the diagnose results
  Rcpp::NumericVector hist_objs, rel_objs;
  
  // initial value of loss function
  double obj0{0}, obj{0}, rel_obj{0};
  //obj0 = 0.5 * std::pow(arma::norm(W % (X-Z),"fro"),2);
  //hist_objs.push_back(obj0);
  
  // iterations
  int k = 0;
  while(k<=maxit){
    if(quiet==0) Rprintf("%i th iteration \n",k);
    
    // form Xtilde
    arma::mat Xtilde = W % X + W_c % Z;
    
    // update Z
    Rcpp::List svd_tmp = svds_f(Xtilde,R);
    U = Rcpp::as<arma::mat>(svd_tmp["u"]);
    S = Rcpp::as<arma::vec>(svd_tmp["d"]);
    V = Rcpp::as<arma::mat>(svd_tmp["v"]);
    Z = U * arma::diagmat(S) * V.t();
    
    // new objective value
    obj = 0.5 * std::pow(arma::norm(W % (X-Z),"fro"),2);
    
    // reporting
    if(k !=0) rel_obj = (obj0-obj)/(obj0+1); 
    hist_objs.push_back(obj);
    if(k !=0) rel_objs.push_back(rel_obj);
    
    // stopping checks
    if(k != 0 && rel_obj < tol_obj) break;
    
    // save previous results
    obj0 = obj;
    ++k;
  }
  
  Rcpp::List diagnose = Rcpp::List::create(Rcpp::Named("hist_objs") = hist_objs,
                                           Rcpp::Named("rel_objs")  = rel_objs);
  Rcpp::List result = Rcpp::List::create(Rcpp::Named("U") = U,
                                         Rcpp::Named("S") = S,
                                         Rcpp::Named("V") = V,
                                         Rcpp::Named("iter") = k,
                                         Rcpp::Named("diagnose") = diagnose);
  return result;
}

//' Model selection of a svd model using missing value based CV error.
//' 
//' This function implemented a missing value based CV model selection approach.
//' First, ratio_mis percent elements are randomly selected as missing values. After
//' that a EM-SVD model is constructed to estimate the prediction error.
//' The details of this function can be found in \url{https://arxiv.org/abs/1902.06241}.
//'
//' @inheritParams alpha_estimation
//' @param ratio_mis the propotion of missing values
//'
//' @return This function returns a matrix contains the K-fold 
//' cross validation errors and the number of PCs used for 
//' model selection.
//'
//' @examples
//' \dontrun{svd_CV(X,K=3,Rs=1:15,ratio_mis=0.1,opts=list())}
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix svd_CV(const arma::mat &X,
                           int K, 
                           const Rcpp::NumericVector &Rs,
                           double ratio_mis,
                           Rcpp::List opts){
  // structure to hold results
  int length_Rs = Rs.length(); 
  Rcpp::NumericMatrix cvErrors(length_Rs, K+1);
  
  // number of non-missing elements
  arma::uvec non_NaN_ind_vec = arma::find_finite(X);
  int mn_nonNaN = non_NaN_ind_vec.n_rows;
  
  // K fold cross validation
  // use opts_inner to do warm start
  Rcpp::List opts_inner = opts;
  
  for(int k=0; k < K; ++k){
    // seperate the Xtest and Xtrain
    // taken into account the potential problem of NaN
    arma::uvec shuf_ind_vec = arma::shuffle(non_NaN_ind_vec);
    arma::uvec index_X_test = shuf_ind_vec.rows(1,std::round(ratio_mis*mn_nonNaN));
    arma::mat X_train = X;
    (X_train(index_X_test)).fill(arma::datum::nan);
    arma::vec X_test = X(index_X_test);
    
    // for loop
    for(int j = (length_Rs-1); j>=0; --j){
      int R = Rs(j);
      
      // using the remaining data to construct a SVD model
      Rcpp::List trainModel = svd_mis(X_train,R,opts_inner);
      arma::mat U = trainModel["U"];
      arma::vec S = trainModel["S"];
      arma::mat V = trainModel["V"];
      arma::mat ZHat = U * diagmat(S) * V.t();
      
      // warm start
      opts_inner["U0"] = U;
      opts_inner["S0"] = S;
      opts_inner["V0"] = V;
      
      // extract the estimated parameters for the prediction of missing elements
      arma::vec X_pred = ZHat(index_X_test);
      
      // compute the prediction error
      cvErrors(j,k) = 0.5 * std::pow(arma::norm(X_test-X_pred,2),2);
      cvErrors(j,K) = R;
    }
  }
  
  Rcpp::CharacterVector col_names(4);
  for(int k=0; k<K; ++k){
    col_names(k) = std::to_string((k+1)) + " th cv";
  }
  col_names(3) = "R";
  Rcpp::colnames(cvErrors) = col_names;

  return cvErrors;

}