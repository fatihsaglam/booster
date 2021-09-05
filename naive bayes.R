naive_bayes <- function(x_train, y_train, w = NULL){
  n_train <- nrow(x_train)
  p <- ncol(x_train)

  if (is.null(w)) {
    w <- rep(1, n_train)
  }
  w <- w*n_train/sum(w)
  
  class_names <- unique(y)
  k_classes <- length(class_names)
  
  n_train <- nrow(x_train)
  n_classes <- sapply(class_names, function(m) sum(y_train == m))
  
  priors <- sapply(class_names, function(m) sum(w[y_train == m])/n_train)

  x_classes <- lapply(class_names, function(m) x_train[y_train == m,])
  w_classes <- lapply(class_names, function(m) w[y_train == m])
  
  means <- lapply(1:k_classes, function(m2) sapply(1:p, function(m) {
    ww <- w_classes[[m2]]/sum(w_classes[[m2]])*n_classes[m2]
    ms <- Hmisc::wtd.mean(x = x_classes[[m2]][,m], na.rm = TRUE, weights = ww)
    return(ms)
  }))
  
  stds <- lapply(1:k_classes, function(m2) sapply(1:p, function(m) {
    ww <- w_classes[[m2]]/sum(w_classes[[m2]])*n_classes[m2]
    vars <- Hmisc::wtd.var(x = x_classes[[m2]][,m], na.rm = TRUE, weights = ww)
    return(sqrt(vars))
  }))
  
  return(list(n_train = n_train,
              p = p,
              x_classes = x_classes,
              n_classes = n_classes,
              k_classes = k_classes,
              priors = priors,
              class_names = class_names,
              means = means,
              stds = stds))
}

predict_naive_bayes <- function(object, newdata, type = "prob"){
  n_train <- object$n_train
  p <- object$p
  x_classes <- object$x_classes
  n_classes <- object$n_classes
  k_classes <- object$k_classes
  priors <- object$priors
  class_names <- object$class_names
  means <- object$means
  stds <- object$stds
  
  x_test <- newdata
  n_test <- nrow(x_test)
  
  densities <- lapply(1:k_classes, function(m) sapply(1:p, function(m2) { 
    d <- dnorm(x_test[,m2], mean = means[[m]][m2], sd = stds[[m]][m2])
    d[is.infinite(d)] <- .Machine$double.xmax
    d[d == 0] <- 1e-20
    return(d)
    }))
  
  likelihoods <- sapply(1:k_classes, function(m) apply(densities[[m]], 1, prod))
  posteriors <- sapply(1:k_classes, function(m) apply(cbind(priors[m], likelihoods[,m]), 1, prod))
  
  posteriors <- t(apply(posteriors, 1, function(m) {
    if(all(m == 0)){
      runif(k_classes, min = 0, max = 1)
    } else{
      m
    }
  }))
  
  posteriors[is.infinite(posteriors)] <- .Machine$double.xmax
  posteriors <- posteriors/apply(posteriors, 1, sum)
  
  colnames(posteriors) <- class_names
  
  if (type == "prob") {
    return(posteriors)
  }
  if (type == "pred") {
    predictions <- apply(posteriors, 1, function(m) class_names[which.max(m)])
    return(predictions)
  }
}
