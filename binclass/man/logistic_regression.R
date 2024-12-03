#' Logistic Regression Using Numerical Optimization
#'
#' Perform logistic regression to estimate the coefficient vector (\eqn{\beta})
#' using numerical optimization of the negative log-likelihood function.
#'
#' @param X A numeric matrix of predictors (dimensions: n x p).
#' @param y A numeric vector of responses (length: n). Values must be 0 or 1.
#' @param tol A numeric value for the convergence tolerance. Default is 1e-6.
#' @param max_iter An integer specifying the maximum number of iterations for optimization. Default is 100.
#' @return A numeric vector of estimated coefficients (\eqn{\beta}).
#' @examples
#' # Simulate data
#' set.seed(123)
#' X <- cbind(1, matrix(rnorm(100), ncol = 2)) # Add intercept
#' y <- rbinom(50, size = 1, prob = 0.5)
#' beta <- logistic_regression(X, y)
#' print(beta)
#' @export
logistic_regression <- function(X, y, tol = 1e-6, max_iter = 100) {
  # X: matrix of predictors (n x p)
  # y: response vector (n x 1)
  # tol: tolerance for convergence
  # max_iter: maximum iterations for the optimization

  # Initial values for beta from least squares estimate
  X_t <- t(X)
  beta_init <- solve(X_t %*% X) %*% X_t %*% y

  # Sigmoid function
  sigmoid <- function(x) {
    1 / (1 + exp(-x))
  }

  # Negative log-likelihood function
  nll <- function(beta) {
    p <- sigmoid(X %*% beta)
    -sum(y * log(p) + (1 - y) * log(1 - p))
  }

  # Gradient of the negative log-likelihood
  gradient <- function(beta) {
    p <- sigmoid(X %*% beta)
    grad <- t(X) %*% (y - p)
    return(grad)
  }

  # Use optim() function for numerical optimization
  result <- optim(beta_init, nll, gr = gradient, method = "BFGS", control = list(maxit = max_iter, reltol = tol))

  return(result$par) # The estimated beta
}
