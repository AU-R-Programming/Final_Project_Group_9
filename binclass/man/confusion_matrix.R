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
