#ifndef CONCAVE_PENALTIES_H
#define CONCAVE_PENALTIES_H

#include <RcppArmadillo.h>
#include<iostream>
#include<limits>
#include<math.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[RcPP::plugins("cpp11")]]

// gdp penalty and its gradient
inline arma::vec gdp(const arma::vec &x, 
              double gamma, double lambda);
inline arma::vec gdp_sg(const arma::vec &x, 
                  double gamma, double lambda);

// lq penalty and its gradient
inline arma::vec lq(const arma::vec &x,
             double gamma, double lambda);
inline arma::vec lq_sg(const arma::vec &x, 
                 double gamma, double lambda);

// SCAD penalty and its gradient
arma::vec scad(const arma::vec &x,
               double gamma, double lambda);
arma::vec scad_sg(const arma::vec &x,
                  double gamma, double lambda);

#endif

