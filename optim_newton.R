optim_newton <- function (training_set, training_lbls,theta,
                                   lambda, maxiters){
    
    alpha    <- 0.1
    found    <- FALSE
    epsilon  <- 10^-4
    prev_theta <- theta
    i <-1 
    cost_vec = vector(length=maxiters)
    
    while ({!found} && {i<maxiters}){
        cost_vec[i] = logistic_cost(theta, training_set, training_lbls, lambda)
        grad <- logistic_grad(theta, training_set, training_lbls, lambda)
        theta <- theta - alpha * grad
        
        #Stop criterion and learning rate control (alpha)
        if(i>1){
            difference <- cost_vec[i-1]-cost_vec[i]
            if (difference >= 0 && difference < epsilon)
                found <- TRUE
            else if (difference < 0){
                theta <- prev_theta
                alpha <- alpha/2
            }
                
            
        }
        i <- i+1
        prev_theta <- theta
    }
    
    res <- list(par=theta,value=cost_vec)
    res
}