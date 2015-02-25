source("split_data.R")
source("train_logistic.R")
source("logistic_cost.R")
source("logistic_grad.R")
source("normalize_data.R")
source("classify_logistic.R")
source("compute_error.R")

n_data <- 1000
#Read Test and Training set. Split data also to dev set
train_f <- read.csv("train.csv", colClasses = "numeric", nrow = n_data)
test_f <- read.csv("test.csv", colClasses = "numeric", nrow = n_data)

#dev_prop <- 0.3 

splitted_data <- split_data(train_f)
training_set  <- splitted_data[[1]]
training_lbls <- splitted_data[[2]] 
dev_set       <- splitted_data[[3]]
dev_lbls      <- splitted_data[[4]]
n_lbls        <- 10 

#Modify labels to become 0's and 1's vector
eye <- diag(n_lbls)
training_lbls_vec <- eye[training_lbls,]
#dev_lbls_vec      <- eye[training_lbls,]

#Normalize training and dev data 
epsilon <- 10^-10
res_lst      <- normalize_data(training_set, epsilon)
training_set <- res_lst[[1]]
data_mean    <- res_lst[[2]]
data_sd      <- res_lst[[3]]


dev_set      <- normalize_data(dev_set, epsilon,mean=data_mean,s_dev=data_sd)[[1]]


#Load test set

splitted_data <- split_data(test_f,dev_prop = 0)
test_set      <- splitted_data[[1]]
test_lbls     <- splitted_data[[2]]

#normalize test data
test_set <- normalize_data(test_set,epsilon,data_mean,data_sd)[[1]]


#Expand data by adding a column of 1's

ones_tr  <- matrix(1,dim(training_set)[1],1)
ones_dev <- matrix(1,dim(dev_set)[1],1)
ones_test <- matrix(1,dim(test_set)[1],1)

training_set <- cbind(ones_tr,training_set)
dev_set      <- cbind(ones_dev ,dev_set)
test_set     <- cbind(ones_test ,test_set)

lambdas <- 0.1 #c(0.001,0.01,0.05,0.1,0.5,1)
#train all logistic models regression model
logistic_models <- matrix(0,nrow=dim(training_set)[2],ncol=n_lbls)

for (i in seq(n_lbls)){
    #print("estoy antes")
    model_param <- train_logistic(training_set, training_lbls_vec[,i],
                                        dev_set, dev_lbls, lambdas)
    logistic_models[,i] <- model_param
    #print(dim(logistic_models))
    
}

classified_res <- classify_logistic(dev_set, logistic_models)
error_rate <- compute_error(dev_lbls,classified_res)