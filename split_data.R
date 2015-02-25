split_data <- function(data, dev_prop=0.3){
    #Splits data to training set and dev set
        
    n <- dim(data)[1]
    n_dev <- floor(dev_prop * n)
    
    # n_train <- n-n_dev
    
    features <- dim(data)[2]
    
    rand_perm <- sample(n) 
    
    if (dev_prop != 0){
        dev_set  <- data[rand_perm[1:n_dev] , 2:features]
        dev_lbls <- data[rand_perm[1:n_dev] , 1] + 1
    }
    
    training_set <- data[rand_perm[(n_dev+1):n],2:features]
    training_lbls <- data[rand_perm[(n_dev+1):n],1] + 1 
    
    if (dev_prop == 0)
        splitted_data <- list(training_set, training_lbls)
    else  
        splitted_data <- list(training_set, training_lbls, dev_set, dev_lbls)
    splitted_data
   
}