source("logistic_cost.R")
source("logistic_grad.R")
source("classify_logistic.R")
source("compute_error.R")
source("optim_newton.R")
library("optimx")

train_logistic <- function(training_set, training_lbls, dev_set=NULL, dev_lbls, lambdas
                           ,alphas=NULL,newton=FALSE, maxiters = 50){
    ## Improve to train with all the dataset !!
    
    
    
    ## Initializing Logistic Model
    features  <- dim(training_set)[2]
    

    low  <-  -10^-4
    high <-   10^-4
    theta    <- runif(features, low, high)
    
    #Call minfunc 
    
    optimal_params <- NULL
    best_error_rate <- Inf
    best_lambda <- NULL
    
    for (lambda in lambdas){
        
        if (newton)
            optim_res <- optim_newton(training_set, training_lbls, 
                                      theta, lambda, maxiters)
        
        else
            optim_res <- optim(par=theta, fn=logistic_cost, gr=logistic_grad, 
                         X=training_set, y=training_lbls,
                         lambda=lambda, method="L-BFGS-B")
                    
        
        # set lambda taking into consideration the Dev_set error rate!
        
        if(!is.null(dev_set)){
        
            classified_lbls <- classify_logistic(dev_set,optim_res$par)
            error_rate <- compute_error(dev_lbls,classified_lbls)
            
            if (error_rate < best_error_rate){
                optimal_params <- optim_res$par
                best_lambda    <- lambda 
            }
        }
        else 
            best_lambda <- lambdas
    }
    #putting as thr first row the lambda value
    optimal_params <- rbind(best_lambda,optimal_params)
    optimal_params
}