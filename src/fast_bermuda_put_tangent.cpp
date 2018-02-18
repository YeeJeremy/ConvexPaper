// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Value function approximation using tangents
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <rflann.h>

// Generate the conditional expectation matrices
arma::cube ExpectMatTangent(const arma::vec& grid,
                            const arma::vec& disturb,
                            const arma::vec& weight) {
  const std::size_t n_grid = grid.n_elem;
  const std::size_t n_disturb = disturb.n_elem;
  // Disturbed grids and nearest neighbours
  arma::uvec neighbour(n_grid * n_disturb);
  {
    arma::mat d_grid(n_grid * n_disturb, 1);
    for (std::size_t dd = 0; dd < n_disturb; dd++) {
      d_grid.rows(n_grid * dd, n_grid * (dd + 1) - 1) = disturb(dd) * grid;
    }
    neighbour = arma::conv_to<arma::uvec>::from(rflann::FastKDNeighbour(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(d_grid)),
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)), 1)) - 1;
  }
  // Conditional expectation operator
  arma::cube perm(n_grid, n_grid, 2, arma::fill::zeros);
  for (std::size_t dd = 0; dd < n_disturb; dd++) {
    for (std::size_t gg = 0; gg < n_grid; gg++) {
      perm(gg, neighbour(n_grid * dd + gg), 0) += weight(dd);
      perm(gg, neighbour(n_grid * dd + gg), 1) += weight(dd) * disturb(dd);
    }
  }
  return(perm);
}

// Bellman recursion using the conditional expectation matrices
//[[Rcpp::export]]
Rcpp::List FastBermudaPutTangent(const double& strike,
                                 const double& discount,
                                 const std::size_t& n_dec,
                                 const arma::vec& grid,
                                 const arma::vec& disturb,
                                 const arma::vec& weight) {
  // Parameters
  const std::size_t n_grid = grid.n_elem;
  const std::size_t n_disturb = disturb.n_elem;
  // Construct the conditional expectation matrices
  const arma::cube perm = ExpectMatTangent(grid, disturb, weight);
  // Bellman recursion
  arma::cube value(n_grid, 2, n_dec, arma::fill::zeros);
  arma::cube cont(n_grid, 2, n_dec - 1, arma::fill::zeros);
  arma::vec cont_value(n_grid);
  // Find in the money paths
  std::size_t in_money = 0;
  for (std::size_t gg = 0; gg < n_grid; gg++) {
    if (grid(gg) >= strike) {
      break;
    }
    in_money++;
  }
  // Initialise with terminal time
  Rcpp::Rcout << "At dec: " << n_dec - 1 << "...";
  value(arma::span(0, in_money), arma::span(0), arma::span(n_dec - 1)) += strike;
  value(arma::span(0, in_money), arma::span(1), arma::span(n_dec - 1)) += -1.;
  // Backward induction
  for (int tt = (n_dec - 2); tt >= 0; tt--) {
    Rcpp::Rcout << tt << ".";
    // Approximating the continuation value
    cont.slice(tt).col(0) = perm.slice(0) * value.slice(tt + 1).col(0);
    cont.slice(tt).col(1) = perm.slice(1) * value.slice(tt + 1).col(1);
    cont.slice(tt) = discount * cont.slice(tt);  // Discounting
    Rcpp::Rcout << ".";
    // Optimise to find value function
    value.slice(tt) = cont.slice(tt);
    cont_value = cont.slice(tt).col(0) + cont.slice(tt).col(1) % grid;
    for (std::size_t gg = 0; gg < in_money; gg++) {
      if (cont_value(gg) <= (strike - grid(gg))) {
        value(gg, 0, tt) = strike;
        value(gg, 1, tt) = -1.;
      }
    }
    for (std::size_t gg = in_money; gg < n_grid; gg++) {
      if (cont_value(gg) <= 0.) {
        value(gg, 0, tt) = 0.;
        value(gg, 1, tt) = 0.;
      }
    }
  }
  return Rcpp::List::create(Rcpp::Named("value") = value,
                            Rcpp::Named("expected") = cont);
}
