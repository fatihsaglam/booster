booster <- function(x_train, 
                    y_train, 
                    classifier = NULL,
                    predictor = NULL,
                    x_test = NULL, 
                    y_test = NULL,
                    weighted_bootstrap = TRUE,
                    max_iter = 30, 
                    alpha_multiplier = 1,
                    print_detail = TRUE,
                    print_plot = FALSE,
                    bag_frac = 0.5) {
  
  n_train <- nrow(x_train)
  p <- ncol(x_train)
  
  class_names <- unique(y_train)
  k_classes <- length(class_names)
  
  n_classes <- sapply(class_names, function(m) sum(y_train == m))
  x_classes <- lapply(class_names, function(m) x_train[y_train == m,])
  
  if (!is.null(x_test) & !is.null(y_test)) {
    n_test <- nrow(x_test)
    err_test <- numeric(max_iter)
    y_test_num <- c(-1,1)[as.numeric(y_test)]
    fit_test <- numeric(n_test)
  }
  
  if (is.null(classifier) &  is.null(predictor)) {
    classifier <- function(x_train, y_train, weights) {
      model <- rpart(formula = y_train~., 
                     data = data.frame(x_train, y_train),
                     weights = weights,
                     control = rpart.control(minsplit = -1,
                                             maxcompete = 0,
                                             maxsurrogate = 0,
                                             usesurrogate = 0))
      return(model)
    }
    
    predictor <- function(model, x_new) {
      x_new <- as.data.frame(x_new)
      preds <- predict(object = model, newdata = x_new, type = "class")
      return(preds)
    }
  }
  
  if (weighted_bootstrap) {
    sampler <- function(w) {
      forced_i <- c(sapply(1:k_classes, function(m) sample(which(y_train == class_names[m]), 2)))
      return(c(sample(x = setdiff(1:n_train, forced_i), 
                      size = 0.632*n_train - k_classes, 
                      replace = TRUE,
                      prob = w[setdiff(1:n_train, forced_i)]), 
               forced_i))
    }
  } else {
    sampler <- function(w) {
      forced_i <- c(sapply(1:k_classes, function(m) sample(which(y_train == class_names[m]), 2)))
      return(c(sample(x = setdiff(1:n_train, forced_i), 
                      size = n_train - n_selected - 2*k_classes, 
                      replace = FALSE), 
               forced_i))
    }
  }
  
  w <- rep(1/n_train, n_train)
  err <- c()
  err_train <- c()
  alpha <- c()
  n_selected <- floor((1 - bag_frac) * n_train)
  models <- list()
  
  fit_train <- matrix(0, nrow = n_train, ncol = k_classes)
  if (!is.null(x_test) & !is.null(y_test)) {
    fit_test <- matrix(0, nrow = n_test, ncol = k_classes)
  }
  
  for (i in 1:max_iter) {
    selection_i <- sampler(w)
    
    x_temp <- x_train[selection_i,]
    y_temp <- y_train[selection_i]
    w_temp <- w[selection_i]
    n_temp <- nrow(x_temp)
    
    models[[i]] <- classifier(x_train = x_temp, y_train = y_temp, weights = w_temp*n_temp/sum(w_temp))
    preds <- predictor(model = models[[i]], x_new = x_train)
    
    err[i] <- sum(w*(preds != y_train))/sum(w)
    if ((1 - err[i]) == 1 | err[i] == 1) {
      err[i] <- (1 - err[i]) * 1e-04 + err[i] * 0.9999
    }

    alpha[i] <- alpha_multiplier * 0.5 * log((1 - err[i])/err[i]) + log(k_classes - 1)
    preds_num <- (sapply(class_names, function(m) as.numeric(preds == m)))
    
    fit_train <- fit_train + alpha[i]*preds_num
    fit_train_pred <- class_names[apply(fit_train, 1, which.max)]
    
    w <- w*exp(alpha[i]*(preds != y_train))
    w <- w/sum(w)
    
    err_train[i] <- sum(fit_train_pred != y_train)/n_train
    
    if ((is.null(x_test) | is.null(y_test)) & print_detail) {
      cat(i, " Train err:", err_train[i], ", Weighted err:", err[i], "\n", sep = "")
      next
    }
    
    preds <- predictor(model = models[[i]], x_new = x_test)
    preds_num <- (sapply(class_names, function(m) as.numeric(preds == m)))
    fit_test <- fit_test + alpha[i]*preds_num
    fit_test_pred <- class_names[apply(fit_test, 1, which.max)]
    err_test[i] <- sum(fit_test_pred != y_test)/n_test
    
    if (print_detail) {
      cat(i, " Train err:", err_train[i], ", Test err:", err_test[i], ", Weighted err:", err[i], "\n", sep = "")
    }
  }
  
  if (print_plot) {
    if (!is.null(x_test) & !is.null(y_test)) {
      plot(err_train, xlab = "Iteration", ylab = "Error", 
           ylim = c(min(c(err_train, err_test)), max(c(err_train, err_test))))
      lines(err_train)
      points(err_test, col = "red", pch = 2)
      lines(err_test, col = "red")
      legend("topright", legend = c("Train", "Test"), lty = c(1,1), col = c("black", "red"), pch = c(1,2))
    } else {
      plot(err_train, xlab = "Iteration", ylab = "Error",
           ylim = c(min(c(err_train)), max(c(err_train))))
      lines(err_train)
      legend("topright", legend = c("Train"), lty = c(1), col = c("black"), pch = c(1))
    }
  }
  
  return(list(n_train = n_train,
              w = w,
              p = p,
              predictor = predictor,
              alpha = alpha,
              models = models,
              x_classes = x_classes,
              n_classes = n_classes,
              k_classes = k_classes,
              class_names = class_names))
}

predict_booster <- function(object, newdata, type = "prob"){
  n_train <- object$n_train
  w <- object$w
  alpha <- object$alpha
  models <- object$models
  x_classes <- object$x_classes
  n_classes <- object$n_classes
  k_classes <- object$k_classes
  class_names <- object$class_names
  predictor <- object$predictor
  
  x_test <- newdata
  n_test <- nrow(x_test)
  
  posteriors_all <- matrix(NA, nrow = n_test, ncol = k_classes)
  
  fit_test <- matrix(0, nrow = n_test, ncol = k_classes)
  
  for (i in 1:length(models)) {
    preds <- predictor(model = models[[i]], x_new = x_test)
    preds_num <- (sapply(class_names, function(m) as.numeric(preds == m)))
    
    fit_test <- fit_test + alpha[i]*preds_num
    print(i)
  }
  
  posteriors <- t(apply(fit_test, 1, function(m) m/sum(m)))
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
