#' Bootstrapped Confidence Intervals for Logistic Regression Coefficients
#'
#' Compute bootstrapped confidence intervals for the coefficients estimated
#' from logistic regression.
#'
#' @param X A numeric matrix of predictors (dimensions: n x p).
#' @param y A numeric vector of responses (length: n). Values must be 0 or 1.
#' @param n_bootstrap An integer specifying the number of bootstrap samples. Default is 20.
#' @param alpha A numeric value specifying the significance level. Default is 0.05.
#' @return A data frame with lower and upper bounds of the confidence intervals.
#' @examples
#' # Simulate data
#' set.seed(123)
#' X <- cbind(1, matrix(rnorm(100), ncol = 2)) # Add intercept
#' y <- rbinom(50, size = 1, prob = 0.5)
#' ci <- bootstrap_CI(X, y, n_bootstrap = 100, alpha = 0.05)
#' print(ci)
#' @export
bootstrap_CI <- function(X, y, n_bootstrap = 20, alpha = 0.05) {
  n <- nrow(X)
  boot_betas <- matrix(NA, nrow = n_bootstrap, ncol = ncol(X))

  for (i in 1:n_bootstrap) {
    sample_indices <- sample(1:n, replace = TRUE)
    X_boot <- X[sample_indices, , drop = FALSE]
    y_boot <- y[sample_indices]

    boot_betas[i, ] <- logistic_regression(X_boot, y_boot)
  }

  # Calculate the lower and upper percentiles for CI
  lower <- apply(boot_betas, 2, function(x) quantile(x, probs = alpha / 2))
  upper <- apply(boot_betas, 2, function(x) quantile(x, probs = 1 - alpha / 2))

  return(data.frame(Lower = lower, Upper = upper))
}
