# Final Project

## created package FinalProjectGroup9.R

#' @import roxygen2
#' @import boot
#' @import ggplot2

# Implement Core Functionality

## 1. Logistic regression using numerical optimization

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
  X_t <- t(X)
  beta_init <- tryCatch(
    solve(X_t %*% X) %*% X_t %*% y,
    error = function(e) rep(0, ncol(X))
  )
  print(beta_init)  # Check initial beta values

  sigmoid <- function(x) {
    p <- 1 / (1 + exp(-x))
    p <- pmax(p, 1e-10)
    p <- pmin(p, 1 - 1e-10)
    return(p)
  }

  nll <- function(beta) {
    p <- sigmoid(X %*% beta)
    print(p)  # Check predicted probabilities
    -sum(y * log(p) + (1 - y) * log(1 - p))
  }

  gradient <- function(beta) {
    p <- sigmoid(X %*% beta)
    t(X) %*% (y - p)
  }

  result <- optim(
    beta_init, nll, gr = gradient, method = "BFGS",
    control = list(maxit = max_iter, reltol = tol)
  )

  if (result$convergence != 0) warning("Optimization did not converge.")
  return(result$par)
}
## 2. Bootstrapped Confidence Intervals

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

## 3. Confusion Matrix and Evaluation Metrics

#' Confusion Matrix and Classification Metrics
#'
#' Generate a confusion matrix and compute performance metrics for binary classification.
#'
#' @param y_true A numeric vector of true binary response values (length: n).
#' @param y_pred A numeric vector of predicted probabilities (length: n).
#' @param cutoff A numeric value specifying the cutoff for classification. Default is 0.5.
#' @return A list containing the confusion matrix and several performance metrics.
#' @examples
#' # Simulate data
#' set.seed(123)
#' y_true <- rbinom(50, size = 1, prob = 0.5)
#' y_pred <- runif(50)
#' metrics <- confusion_matrix_metrics(y_true, y_pred, cutoff = 0.5)
#' print(metrics)
#' @export
confusion_matrix_metrics <- function(y_true, y_pred, cutoff = 0.5) {
  # y_true: actual response values (0/1)
  # y_pred: predicted probabilities (between 0 and 1)
  # cutoff: threshold for converting predicted probabilities into class labels

  y_pred_class <- ifelse(y_pred > cutoff, 1, 0)

  # Confusion matrix components
  tp <- sum(y_true == 1 & y_pred_class == 1)
  tn <- sum(y_true == 0 & y_pred_class == 0)
  fp <- sum(y_true == 0 & y_pred_class == 1)
  fn <- sum(y_true == 1 & y_pred_class == 0)

  # Compute metrics
  prevalence <- (tp + fn) / length(y_true)
  accuracy <- (tp + tn) / length(y_true)
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  fdr <- fp / (fp + tp)
  dor <- (tp * tn) / (fp * fn)

  return(list(
    ConfusionMatrix = matrix(c(tp, fp, fn, tn), nrow = 2),
    Prevalence = prevalence,
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity,
    FalseDiscoveryRate = fdr,
    DiagnosticOddsRatio = dor
  ))
}
