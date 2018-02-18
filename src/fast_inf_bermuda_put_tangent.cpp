// Copyright 2018 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Value function iteration using tangents
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <rflann.h>

// Generate the conditional expectation matrices
arma::cube ExpectMatTangent2(const arma::vec& grid,
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
Rcpp::List FastInfBermudaPutTangent(const double& strike,
                                    const double& discount,
                                    const arma::vec& grid,
                                    const arma::vec& disturb,
                                    const arma::vec& weight,
                                    const std::size_t& max_iter,
                                    const double& error) {
  // Parameters
  const std::size_t n_grid = grid.n_elem;
  const std::size_t n_disturb = disturb.n_elem;
  // Construct the conditional expectation matrices
  const arma::cube perm = ExpectMatTangent2(grid, disturb, weight);
  // Find in the money paths
  std::size_t in_money = 0;
  for (std::size_t gg = 0; gg < n_grid; gg++) {
    if (grid(gg) >= strike) {
      break;
    }
    in_money++;
  }
  // Initialise with payoff function
  arma::mat value1(n_grid, 2, arma::fill::zeros);
  value1.rows(0, in_money).col(0) += strike;
  value1.rows(0, in_money).col(1) += -1.;
  arma::mat value2 = value1;
  arma::mat cont(n_grid, 2, arma::fill::zeros);
  arma::vec cont_value(n_grid);
  arma::mat compare(n_grid, 2);
  // Value iteration
  double iter_error = 9999;
  std::size_t count = 1;
  while (iter_error > error) {
    if (count > max_iter) {
      break;
    }
    Rcpp::Rcout << count << ".";
    // Approximating the continuation value
    cont.col(0) = perm.slice(0) * value1.col(0);
    cont.col(1) = perm.slice(1) * value1.col(1);
    cont = discount * cont;  // Discounting
    Rcpp::Rcout << ".";
    // Optimise to find value function
    value2 = cont;
    cont_value = cont.col(0) + cont.col(1) % grid;
    for (std::size_t gg = 0; gg < in_money; gg++) {
      if (cont_value(gg) <= (strike - grid(gg))) {
        value2(gg, 0) = strike;
        value2(gg, 1) = -1.;
      }
    }
    for (std::size_t gg = in_money; gg < n_grid; gg++) {
      if (cont_value(gg) <= 0.) {
        value2(gg, 0) = 0.;
        value2(gg, 1) = 0.;
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
