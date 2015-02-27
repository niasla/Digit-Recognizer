source("split_data.R")
source("train_logistic.R")
source("logistic_cost.R")
source("logistic_grad.R")
source("normalize_data.R")
source("classify_logistic.R")
source("compute_error.R")
library("doMC")


#Rprof("profiling.out")
registerDoMC()
n_data <- 100
#Read Test and Training set. Split data also to dev set
train_f <- read.csv("train.csv", colClasses = "numeric", nrow = n_data)


#print(dim(train_f))
#dev_prop <- 0.3 

splitted_data <- split_data(train_f)
training_set  <- splitted_data[[1]]
training_lbls <- splitted_data[[2]] 
dev_set       <- splitted_data[[3]]
dev_lbls      <- splitted_data[[4]]
test_dev_set  <- splitted_data[[5]]
test_dev_lbls <- splitted_data[[6]]
n_lbls        <- 10 

#Modify labels to become 0's and 1's vector
eye <- diag(n_lbls)
training_lbls_vec <- eye[training_lbls,]
#dev_lbls_vec      <- eye[training_lbls,]

#Normalize training, dev data and test_dev
epsilon <- 10^-10
res_lst      <- normalize_data(training_set, epsilon)
training_set <- res_lst[[1]]
data_mean    <- res_lst[[2]]
data_sd      <- res_lst[[3]]


dev_set      <- normalize_data(dev_set, epsilon,mean=data_mean,s_dev=data_sd)[[1]]
test_dev_set <- normalize_data(test_dev_set, epsilon,mean=data_mean,s_dev=data_sd)[[1]]

#Load test set 


#normalize test data
#test_set <- normalize_data(test_set,epsilon,data_mean,data_sd)[[1]]


#Expand data by adding a column of 1's

ones_tr  <- matrix(1,dim(training_set)[1],1)
ones_dev <- matrix(1,dim(dev_set)[1],1)
ones_test <- matrix(1,dim(test_dev_set)[1],1)

training_set <- cbind(ones_tr,training_set)
dev_set      <- cbind(ones_dev ,dev_set)
test_dev_set <- cbind(ones_test ,test_dev_set)

lambdas <-c(0.1,0.5,1,2,3,4,5)
#train all logistic models regression model

logistic_models <- foreach(i= 1:n_lbls,.combine=cbind)%dopar%{
    train_logistic(training_set, training_lbls_vec[,i],
                   dev_set, dev_lbls, lambdas, newton=TRUE,maxiters=100)
}


# Save Model Parameters
save(list(data_mean=data_mean,
          data_sd  =data_sd,
          lambda   =logistic_models[1,],
          theta    =logistic_models[-1,]),
          file = "logistic_model.rda")


classified_res <- classify_logistic(test_dev_set, as.matrix(logistic_models[-1,]))
error_rate <- compute_error(test_dev_lbls,classified_res)
cat(sprintf("Accuracy: %f%%\n",100*(1-error_rate)))
#summaryRprof("profiling.out")