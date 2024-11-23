# Create package structure

#' Binary Classification Model
#'
#' @param X Matrix of predictor variables
#' @param y Vector of binary response (0 or 1)
#' @param alpha Significance level for confidence intervals (default: 0.05)
#' @param n_bootstrap Number of bootstrap iterations (default: 20)
#' @return List containing model results
#' @export
binary_classifier <- function(X, y, alpha = 0.05, n_bootstrap = 20) {
  # Add intercept column to X
  X <- cbind(1, X)

  # Initial values using least squares
  initial_beta <- solve(t(X) %*% X) %*% t(X) %*% y

  # Objective function (negative log-likelihood)
  objective <- function(beta) {
    p <- 1 / (1 + exp(-X %*% beta))
    -sum(y * log(p) + (1 - y) * log(1 - p))
  }

  # Optimize using BFGS
  opt_result <- optim(initial_beta, objective, method = "BFGS")
  beta_hat <- opt_result$par

  # Bootstrap confidence intervals
  bootstrap_betas <- matrix(0, nrow = n_bootstrap, ncol = length(beta_hat))
  n <- length(y)

  for(i in 1:n_bootstrap) {
    idx <- sample(1:n, n, replace = TRUE)
    X_boot <- X[idx,]
    y_boot <- y[idx]

    opt_boot <- optim(initial_beta, function(beta) {
      p <- 1 / (1 + exp(-X_boot %*% beta))
      -sum(y_boot * log(p) + (1 - y_boot) * log(1 - p))
    }, method = "BFGS")

    bootstrap_betas[i,] <- opt_boot$par
  }

  # Calculate confidence intervals
  ci_lower <- apply(bootstrap_betas, 2, function(x) quantile(x, alpha/2))
  ci_upper <- apply(bootstrap_betas, 2, function(x) quantile(x, 1-alpha/2))

  # Predictions
  p_hat <- 1 / (1 + exp(-X %*% beta_hat))
  y_pred <- ifelse(p_hat > 0.5, 1, 0)

  # Confusion matrix
  conf_matrix <- table(Actual = y, Predicted = y_pred)

  # Calculate metrics
  TP <- conf_matrix[2,2]
  TN <- conf_matrix[1,1]
  FP <- conf_matrix[1,2]
  FN <- conf_matrix[2,1]

  prevalence <- mean(y)
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  fdr <- FP / (FP + TP)
  dor <- (TP * TN) / (FP * FN)

  return(list(
    coefficients = beta_hat,
    initial_values = initial_beta,
    confidence_intervals = list(
      lower = ci_lower,
      upper = ci_upper
    ),
    confusion_matrix = conf_matrix,
    metrics = list(
      prevalence = prevalence,
      accuracy = accuracy,
      sensitivity = sensitivity,
      specificity = specificity,
      false_discovery_rate = fdr,
      diagnostic_odds_ratio = dor
    )
  ))
}

# Example usage function
#' @examples
#' X <- matrix(rnorm(100*2), ncol=2)
#' y <- rbinom(100, 1, 0.5)
#' result <- binary_classifier(X, y)
#' print(result$metrics)
