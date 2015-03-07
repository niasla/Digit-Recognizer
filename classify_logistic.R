classify_logistic <- function (test_set, logistic_models){
    
    n_models <- dim(logistic_models)[2];
    
    # Build theta matrix for multiple models, each column is a theta vector.
    #theta = matrix(,nrow = dim(logistic_models[2])[2],ncol = n_models)
    #for i in seq(n_models){
#         theta[,i] = logistic_models[i][2]
#     }
#     
    X_matrix <- test_set
    
    theta <- logistic_models  
    
    h <- 1/{1+exp(-X_matrix%*%theta ) }   
    
    classify_res <- max.col(h)
    
    classify_res
    
}