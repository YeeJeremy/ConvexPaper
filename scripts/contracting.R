library(ConvexPaper)
## Parameters for put option
rate <- 0.15
step <- 0.25
vol <- 0.1
strike <- 40
discount <- exp(-rate * step)
max_iter <- 1000
error <- 0.001
## Grid
n_grid <- 51
grid <- seq(20, 70, length = n_grid)
## Disturbances
nD <- 1000  ## size of sampling
dist <- rep(NA, nD)
u <- (rate - 0.5 * vol^2) * step
sigma <- vol * sqrt(step)
pExpect <- function(a, b){  ## Partial expectation on [a,b]
    aa <- (log(a) - (u + sigma^2)) / sigma
    bb <- (log(b) - (u + sigma^2)) / sigma
    vv <- exp(u + sigma^2/2) * (pnorm(bb) - pnorm(aa))
    return(vv)
}
## Disturbance sampling using conditional averages
part <- qlnorm(seq(0, 1, length = nD + 1), u, sigma)
for (i in 1:nD) {
    dist[i] <- pExpect(part[i], part[i+1]) / (plnorm(part[i+1], u, sigma) - plnorm(part[i], u, sigma))
}
weight <- rep(1/nD, nD)
## Bellman recursion using the tangent approximation
time1 <- proc.time()
bellman1 <- FastInfBermudaPutTangent(strike, discount, grid, dist, weight, max_iter, error)
time1 <- proc.time() - time1
## Disturbance sampling using extreme values
begin <- .0000000005
end <-  .9999999995
a <- 1 / (end - begin)  # alpha
dist <- qlnorm(seq(begin, end, length = nD), u, sigma)
weight[1] <- (dist[2] / (nD - 1) - a * pExpect(dist[1], dist[2])) / (dist[2] - dist[1])
for (i in 2:(nD - 1)) {
    weight[i] <-
        ((dist[i+1] / (nD - 1)) - a * pExpect(dist[i], dist[i+1])) / (dist[i+1] - dist[i]) +
        (a * pExpect(dist[i-1], dist[i]) - (dist[i-1] / (nD - 1))) / (dist[i] - dist[i-1])
}
weight[nD] <- (a * pExpect(dist[nD - 1], dist[nD]) - dist[nD - 1] / (nD - 1)) / (dist[nD] - dist[nD-1])
## Bellman
lipz <- strike - grid[1]
time2 <- proc.time()
bellman2 <- InfBermudaPutUpper(strike, discount, grid, lipz, dist, weight, max_iter, error) 
time2 <- proc.time() - time2
## Output the results
ggIndex <- c(32,34,36,38,40,42,44,46)
lower <- rep(NA, length(ggIndex))
upper <- lower
gap <- lower
## Value functions
value1 <- rowSums(bellman1$value * c(rep(1, n_grid), grid))
value2 <- bellman2$value
for (gg in 1:length(ggIndex)) {
    host <- which(grid == ggIndex[gg])
    lower[gg] <- value1[host]
    upper[gg] <- value2[host]
    gap[gg] <- upper[gg] - lower[gg]
}
## Print results and computational times
print(round(cbind(ggIndex, lower, upper, gap), 5))
print(time1)
print(time2)
print(bellman1$error)
print(bellman2$error)
