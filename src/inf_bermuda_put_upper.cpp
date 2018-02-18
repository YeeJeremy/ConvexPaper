// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Value function approximation using tangents
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <rflann.h>

// Generate the conditional expectation matrices
arma::mat ExpectMatUpper2(const arma::vec& grid,
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
Rcpp::List InfBermudaPutUpper(const double& strike,
                              const double& discount,
                              const arma::vec& grid,
                              const double& lipz,
                              const arma::vec& disturb,
                              const arma::vec& weight,
                              const std::size_t& max_iter,
                              const double& error) {
  
  // Parameters
  const std::size_t n_grid = grid.n_elem;
  const std::size_t n_disturb = disturb.n_elem;
  // Construct the conditional expectation matrices
  const arma::mat perm = ExpectMatUpper2(grid, lipz, disturb, weight);
  // Bellman recursion
  arma::vec value1(n_grid, arma::fill::zeros);
  arma::vec cont(n_grid, arma::fill::zeros);
  // Find in the money paths
  std::size_t in_money = 0;
  for (std::size_t gg = 0; gg < n_grid; gg++) {
    if (grid(gg) >= strike) {
      break;
    }
    in_money++;
  }
  // Initialise with payoff function
  value1(arma::span(0, in_money)) = strike - grid(arma::span(0, in_money));
  arma::vec value2 = value1;
  // Value iteration
  arma::vec compare(n_grid);
  double iter_error = 9999;
  std::size_t count = 1;
  while (iter_error > error) {
    if (count > max_iter) {
      break;
    }
    Rcpp::Rcout << count << ".";
    // Approximating the continuation value
    cont = perm * value1;
    cont = discount * cont;  // Discounting
    Rcpp::Rcout << ".";
    // Optimise to find value function
    value2 = cont;
    for (std::size_t gg = 0; gg < in_money; gg++) {
      if (cont(gg) <= (strike - grid(gg))) {
        value2(gg) = strike - grid(gg);
      }
    }
    for (std::size_t gg = in_money; gg < n_grid; gg++) {
      if (cont(gg) >= value2(in_money - 1)) {
        value2(gg) = value2(in_money - 1);
      }
    }
    compare = arma::abs(value1 - value2);
    iter_error = compare.max();
    value1 = value2;
    count += 1;
  }
  return Rcpp::List::create(Rcpp::Named("value") = value2,
                            Rcpp::Named("expected") = cont,
                            Rcpp::Named("error") = iter_error);
}
