logistic_grad <- function(theta, X, y, lambda){
    m <- dim(X)[1]
    #h <- 1/(1+exp(-t(theta) %*% t(X)))
    
    #X_matrix <- as.matrix(X)
    h <- 1/{1+exp(-X%*%theta ) }
    
    #grad <- (1/m)*(t(X_matrix)%*%(h-y))
    grad <- {1/m}*crossprod(X,h-y)
    
    reg_factor <- as.matrix(grad[-1] + {lambda/m}*grad[-1])
    
    grad <- rbind(grad[1] , reg_factor)
    grad
}