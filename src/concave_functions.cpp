#include <RcppArmadillo.h>
#include <iostream>
#include <limits>
#include <math.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]

// SCAD penalty function
arma::vec scad(const arma::vec &x,
               double gamma, 
			   double lambda){
  // all elments of x should be nonnegative
  if(x.min() < 0)
    throw std::range_error("numeric vector x contains negative elements.");
  
  arma::vec y = lambda * x;
  
  for(int i = 0; i < x.n_elem; ++i){
    if(x[i] > (gamma*lambda)){
      y[i] = 0.5 * (gamma + 1) * std::pow(lambda, 2);
    }else if(x[i] > lambda){
      y[i] = (-std::pow(x[i], 2) + 2*gamma*lambda*x[i] - std::pow(lambda, 2))/(2*(gamma-1));
    }
  }
  
  return y;
}
 
// Super gradient for the SCAD penalty function
arma::vec scad_sg(const arma::vec &x,
                  double gamma, double lambda){
  // all elments of x should be nonnegative
  if(x.min() < 0)
    throw std::range_error("numeric vector x contains negative elements.");
  
  arma::vec y(arma::size(x), arma::fill::ones);
  y = y*lambda;
  
  for(int i = 0; i < x.n_elem; ++i){
    if(x[i] > (gamma*lambda)){
      y[i] = 0;
    }else if(x[i] > lambda){
      y[i] = (gamma*lambda - x[i])/(gamma-1);
    }
  }
  
  return y;
}
