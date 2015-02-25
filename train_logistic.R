source("logistic_cost.R")
source("logistic_grad.R")
source("classify_logistic.R")
source("compute_error.R")

train_logistic <- function(training_set, training_lbls, dev_set, dev_lbls, lambdas){
    ## Improve to train with all the dataset !!
    
    
    
    ## Initializing Logistic Model
    features <- dim(training_set)[2]
    theta    <- matrix(runif(features, 0, 1), nrow=features)
    
    #Call minfunc 
    
    optimal_model <- NULL
    best_error_rate <- Inf
    
    for (lambda in lambdas){
        optim_res <- optim(par=theta, fn=logistic_cost, gr=logistic_grad, 
                     X=training_set, y=training_lbls,lambda=lambda)
        
        # set lambda taking into consideration the Dev_set error rate!
        classified_lbls <- classify_logistic(dev_set,optim_res$par)
        error_rate <- compute_error(dev_lbls,classified_lbls)
        
        if (error_rate < best_error_rate)
            optimal_model <- optim_res$par
    }
    
    optimal_model
}