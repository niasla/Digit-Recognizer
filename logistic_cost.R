logistic_cost <- function(theta, X, y, lambda){
    #h numerical correction to avoid log(0) 
    correction_h <- function(x){
        epsilon <- 10^-5
        if(x == 0){
            res <- x+epsilon
        }
        else if(x == 1){
            res <- x-epsilon
        }
        else     
            res <- x
    
        res
    }
    
    m <- dim(X)[1]    
    #Posibility to switch X %*% theta
    X_matrix <- as.matrix(X)

    h <- 1/(1+exp(-X_matrix%*%as.vector(theta) ) )
    #print(X_matrix%*%theta)
    #print(min(h))
    
    h <- apply(h,c(1,2),correction_h)
    
    
    #print(dim(theta))
    #print(log(1-h))
    J <- (-1/m) * sum( y*log(h) + (1-y)*log(1-h) )
    theta_squared <- theta[-1]^2
    theta_squared <- rbind(theta[1],theta_squared) 
    J <- J + (lambda/(2*m)) * sum(theta_squared)
    J
    #print(J)
}