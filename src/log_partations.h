#ifndef LOG_PARTATIONS_H
#define LOG_PARTATIONS_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' The log-partation function of Bernoulli distribution
//'
//' This function define the log-partation function
//' of Bernoulli distribution. The formula is
//' \code{log_part_B(Theta) = log(1+exp(Theta))}.
//' Details can be found in \url{https://arxiv.org/abs/1902.06241}.
//'
//' @param Theta a matrix of natural parameter
//'
//' @return The value of the corresponding log-partation function
//'
//' @examples
//' \dontrun{log_part_B(matrix(0:9,2,5))}
inline arma::mat log_part_B(const arma::mat &Theta){
  return arma::log(1 + arma::exp(Theta));
}

//' The first order gradient of log-partation function of Bernoulli distribution
//'
//' This function define first order gradient of the log-partation function
//' of Bernoulli distribution. The formula is
//' \code{log_part_B_g(Theta) = exp(Theta)/(1+exp(Theta))}.
//' Details can be found in \url{https://arxiv.org/abs/1902.06241}.
//'
//' @inheritParams log_part_B
//'
//' @return value of the first order gradient
//'
//' @examples
//' \dontrun{log_part_B_g(matrix(0:9,2,5))}
inline arma::mat log_part_B_g(const arma::mat &Theta){
  
  return  arma::exp(Theta)/(1 + arma::exp(Theta));
}

//' The log-partation function of Gaussian distribution
//'
//' This function define the log-partation function
//' of Gaussian distribution. The formula is
//' \code{log_part_G(Theta) = 0.5*(Theta^2)}.
//' Details can be found in \url{https://arxiv.org/abs/1902.06241}.
//'
//' @inheritParams log_part_B
//'
//' @return the value of the corresponding log-partation function
//'
//' @examples
//' \dontrun{log_part_G(matrix(0:9,2,5))}
inline arma::mat log_part_G(const arma::mat &Theta){
  
  return 0.5 * arma::pow(Theta, 2);
}

//' The first order gradient of log-partation function of Gaussian distribution
//'
//' This function define first order gradient of the log-partation function
//' of Gaussian distribution. The formula is
//' \code{log_part_G_g(Theta) = Theta}.
//' Details can be found in \url{https://arxiv.org/abs/1902.06241}.
//'
//' @inheritParams log_part_G
//'
//' @return value of the first order gradient
//'
//' @examples
//' \dontrun{log_part_G_g(matrix(0:9,2,5))}
inline arma::mat log_part_G_g(const arma::mat &Theta){
  
  return Theta;
}

//' The log-partation function of Possion distribution
//'
//' This function define the log-partation function
//' of Possion distribution. The formula is
//' \code{log_part_G(Theta) = exp(Theta)}.
//' Details can be found in \url{https://arxiv.org/abs/1902.06241}.
//'
//' @inheritParams log_part_B
//'
//' @return the value of the corresponding log-partation function
//'
//' @examples
//' \dontrun{log_part_P(matrix(0:9,2,5))}
inline arma::mat log_part_P(const arma::mat &Theta){
  
  return arma::exp(Theta);
}

//' The first order gradient of log-partation function of Possion distribution
//'
//' This function define first order gradient of the log-partation function
//' of Possion distribution. The formula is
//' \code{log_part_P_g(Theta) = exp(Theta)}.
//' Details can be found in \url{https://arxiv.org/abs/1902.06241}.
//'
//' @inheritParams log_part_P
//'
//' @return value of the first order gradient
//'
//' @examples
//' \dontrun{log_part_P_g(matrix(0:9,2,5))}
inline arma::mat log_part_P_g(const arma::mat &Theta){
  
  return arma::exp(Theta);
}

#endif
