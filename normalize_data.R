normalize_data <- function(data, epsilon=0, mean=NULL, s_dev=NULL){
    if (is.null(mean))
        data_mean <- colMeans(data)
    else
        data_mean <- mean
        
    numerator_norm <- t(apply(data,1,'-',data_mean))
    
    if(is.null(s_dev))
        standard_dev <- apply(data,2,sd) + epsilon   # To avoid Nan's
    else
        standard_dev <- s_dev
    
    data_norm <- t(apply(numerator_norm,1,'/', standard_dev))
    res <- list(data_norm, data_mean, standard_dev)
    
}