// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Value function approximation using tangents
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <rflann.h>

// Generate the conditional expectation matrices
arma::mat ExpectMatUpper(const arma::vec& grid,
                         const double& lipz,
                         const arma::vec& disturb,
                         const arma::vec& weight) {
  const std::size_t n_grid = grid.n_elem;
  const std::size_t n_disturb = disturb.n_elem;
  // Disturbed grids and nearest neighbours
  arma::umat neighbour(n_grid * n_disturb, 2);
  arma::mat d_grid(n_grid * n_disturb, 1);
  for (std::size_t dd = 0; dd < n_disturb; dd++) {
    d_grid.rows(n_grid * dd, n_grid * (dd + 1) - 1) = disturb(dd) * grid;
  }
  neighbour = arma::conv_to<arma::umat>::from(rflann::FastKDNeighbour(
      Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(d_grid)),
      Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)), 2)) - 1;
  // Conditional expectation operator
  arma::uword ii;
  arma::uword host1;
  arma::uword host2;
  arma::mat perm(n_grid, n_grid, arma::fill::zeros);
  for (std::size_t dd = 0; dd < n_disturb; dd++) {
    for (std::size_t gg = 0; gg < n_grid; gg++) {
      ii = n_grid * dd + gg;
      if (d_grid(ii) <= grid(0)) {
        perm(gg, 0) += weight(dd) * (lipz + (grid(0) - d_grid(ii))) / lipz;
      } else if (d_grid(ii) >= grid(n_grid - 1)) {
        perm(gg, n_grid - 1) += weight(dd);
      } else {
        host1 = neighbour(ii, 0);
        host2 = neighbour(ii, 1);
        perm(gg, host1) += weight(dd) * std::abs(d_grid(ii) - grid(host2))
            / std::abs(grid(host1) - grid(host2));
        perm(gg, host2) += weight(dd) * std::abs(d_grid(ii) - grid(host1))
            / std::abs(grid(host1) - grid(host2));
      }
    }
  }
  return(perm);
}

// Bellman recursion using the conditional expectation matrices
//[[Rcpp::export]]
Rcpp::List BermudaPutUpper(const double& strike,
                           const double& discount,
                           const std::size_t& n_dec,
                           const arma::vec& grid,
                           const double& lipz,
                           const arma::vec& disturb,
                           const arma::vec& weight) {
  
  // Parameters
  const std::size_t n_grid = grid.n_elem;
  const std::size_t n_disturb = disturb.n_elem;
  // Construct the conditional expectation matrices
  const arma::mat perm = ExpectMatUpper(grid, lipz, disturb, weight);
  // Bellman recursion
  arma::mat value(n_grid, n_dec, arma::fill::zeros);
  arma::mat cont(n_grid, n_dec - 1, arma::fill::zeros);
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
  value(arma::span(0, in_money), arma::span(n_dec - 1)) =
      strike - grid(arma::span(0, in_money));
  // Backward induction
  for (int tt = (n_dec - 2); tt >= 0; tt--) {
    Rcpp::Rcout << tt << ".";
    // Approximating the continuation value
    cont.col(tt) = perm * value.col(tt + 1);
    cont.col(tt) = discount * cont.col(tt);  // Discounting
    Rcpp::Rcout << ".";
    // Optimise to find value function
    value.col(tt) = cont.col(tt);
    for (std::size_t gg = 0; gg < in_money; gg++) {
      if (cont(gg, tt) <= (strike - grid(gg))) {
        value(gg, tt) = strike - grid(gg);
      }
    }
  }
  return Rcpp::List::create(Rcpp::Named("value") = value,
                            Rcpp::Named("expected") = cont);
}
