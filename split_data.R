split_data <- function(data, dev_prop=0.2,test_prop=0.2){
    #Splits data to training set and dev set
        
    n <- dim(data)[1]
    n_dev <- floor(dev_prop * n)
    n_test <- floor(test_prop * n)
    
    # n_train <- n-n_dev
    
    features <- dim(data)[2]
    
    rand_perm <- sample(n) 
    
    initial_idx = 1;
    if (dev_prop != 0){
        dev_set  <- data[rand_perm[initial_idx:n_dev] , 2:features]
        dev_lbls <- data[rand_perm[initial_idx:n_dev] , 1] + 1
        initial_idx =  n_dev + 1

        
        test_set  <- data[rand_perm[initial_idx:(initial_idx+n_test)],2:features]
        test_lbls <- data[rand_perm[initial_idx:(initial_idx+n_test)],1] + 1  
        initial_idx = initial_idx + n_test +1 ;
    }
    #print(initial_idx)
   
    training_set  <- data[rand_perm[initial_idx:n],2:features]
    training_lbls <- data[rand_perm[initial_idx:n],1] + 1 
    
    
    if (dev_prop == 0)
        splitted_data <- list(training_set, training_lbls)
    else  
        splitted_data <- list(training_set, training_lbls, dev_set,
                            dev_lbls,test_set, test_lbls)
   
    splitted_data
   
}