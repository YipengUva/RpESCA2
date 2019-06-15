#ifndef CONCAVE_FUNCTIONS_H
#define CONCAVE_FUNCTIONS_H

#include <RcppArmadillo.h>
#include <iostream>
#include <limits>
#include <math.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]

/*
#' gdp penalty function
#'
#' This function define the gdp penalty. The
#' formula is \code{gdp(x) =lambda*log(1+x/gamma)}.
#' Details can be found in \url{https://arxiv.org/abs/1807.04982}.
#'
#' @param x a non-negative numeric vector
#' @param gamma hyper-tuning parameter
#' @param lambda tuning parameter
#'
#' @return the value of gdp penalty
#'
#' @examples
#' \dontrun{gdp(0:9,gamma=1,lambda=5)}
*/
inline arma::vec gdp(const arma::vec &x, 
                     double gamma, double lambda){
  
  // all elments of x should be nonnegative
  if(x.min() < 0)
    throw std::range_error("numeric vector x contains negative elements.");
  
  return lambda * arma::log(1+x/gamma);
}


/*
#' Super gradient of gdp penalty function
#'
#' This function define the super gradient of the gdp penalty.
#' The formula is \code{gdp_sg(x) = lambda/(gamma+x)}.
#' Details can be found in \url{https://arxiv.org/abs/1807.04982}
#'
#' @inheritParams gdp
#'
#' @return the super-gradient of the gdp penalty function
#'
#' @examples
#' \dontrun{gdp_sg(0:9,gamma=1,lambda=5)}
*/
inline arma::vec gdp_sg(const arma::vec &x, 
                        double gamma, double lambda){
  
  // all elments of x should be nonnegative
  if(x.min() < 0)
    throw std::range_error("numeric vector x contains negative elements.");
  
  return lambda/(gamma+x);
}

/*
#' Lq penalty function
#'
#' This function define the Lq penalty. The
#' formula is \code{Lq(x) =lambda*x^q}.
#' Details can be found in \url{https://arxiv.org/abs/1807.04982}
#'
#' @inheritParams gdp
#'
#' @return the value of the Lq penalty
#'
#' @examples
#' \dontrun{lq(0:9,gamma=0.5,lambda=5)}
*/
inline arma::vec lq(const arma::vec &x,
                    double gamma, double lambda){
  // all elments of x should be nonnegative
  if(x.min() < 0)
    throw std::range_error("numeric vector x contains negative elements.");
  
  double epsilon = 0.5 * std::numeric_limits<double>::epsilon();
  
  return lambda * arma::pow(x+epsilon, gamma);
}

/*
#' Super gradient of the Lq penalty function
#'
#' This function define the super gradient of the Lq penalty.
#' The formula is \code{lq_sg(x) = lambda * q * x^(p-1)}.
#' Details can be found in \url{https://arxiv.org/abs/1807.04982}
#'
#' @inheritParams lq
#'
#' @return the super-gradient of the Lq penalty function
#'
#' @examples
#' \dontrun{
#' lq_sg(0:9,gamma=1,lambda=5)
#' lq_sg(0:9,gamma=0.5,lambda=5)
#' }
*/
inline arma::vec lq_sg(const arma::vec &x, 
                       double gamma, double lambda){
  // all elments of x should be nonnegative
  if(x.min() < 0)
    throw std::range_error("numeric vector x contains negative elements.");
  
  double epsilon = 0.5*std::numeric_limits<double>::epsilon();
  
  return lambda * gamma * arma::pow(x+epsilon, gamma-1);
}

/* 
#' SCAD penalty function
#'
#' This function define the SCAD penalty.
#' Details can be found in \url{https://arxiv.org/abs/1807.04982}
#'
#' @inheritParams gdp
#'
#' @return the value of SCAD penalty
#'
#' @examples
#' \dontrun{scad(0:9,gamma=3.7,lambda=3)}
*/
arma::vec scad(const arma::vec &x,
               double gamma, double lambda);
 
/* 
#' Super gradient for the SCAD penalty function
#'
#' This function define the supergradient of the SCAD penalty.
#' Details can be found in \url{https://arxiv.org/abs/1807.04982}
#'
#' @inheritParams scad
#'
#' @return the super gradient of the SCAD penalty
#'
#' @examples
#' \dontrun{scad_sg(0:9,gamma=3.7,lambda=3)}
*/ 
arma::vec scad_sg(const arma::vec &x,
                  double gamma, double lambda);

#endif

