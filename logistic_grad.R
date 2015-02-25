logistic_grad <- function(theta, X, y, lambda){
    m <- dim(X)[1]
    #h <- 1/(1+exp(-t(theta) %*% t(X)))
    
    X_matrix <- as.matrix(X)
    h <- 1/(1+exp(-X_matrix%*%theta ) )
    
    grad <- (1/m)*(t(X_matrix)%*%(h-y))
    
    reg_factor <- grad[-1] + (lambda/m)*grad[-1]
    
    grad <- rbind(grad[1] , as.matrix(reg_factor))
    grad
}