source("split_data.R")
source("train_logistic.R")
source("logistic_cost.R")
source("logistic_grad.R")
source("normalize_data.R")

#Rprof("profiling.out")
registerDoMC()
n_data <- 1000
#Read Test and Training set. Split data also to dev set
train_f <- read.csv("train.csv", colClasses = "numeric", nrow = n_data)
test_f <- read.csv("test.csv", colClasses = "numeric", nrow = n_data)

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

#Ecpand data to ad column of ones
ones_tr  <- matrix(1,dim(training_set)[1],1)
ones_dev <- matrix(1,dim(dev_set)[1],1)
ones_test <- matrix(1,dim(test_dev_set)[1],1)

training_set <- cbind(ones_tr,training_set)
dev_set      <- cbind(ones_dev ,dev_set)
test_dev_set <- cbind(ones_test ,test_dev_set)

# lambdas <-0.1 #c(0.01,0.1,0.5,1,2,3,4,5,8,10)
# #train all logistic models regression model
# #logistic_models <- matrix(0,nrow=dim(training_set)[2],ncol=n_lbls)
# 
# logistic_models <- foreach(i= 1:n_lbls,.combine=cbind)%dopar%{
#     train_logistic(training_set, training_lbls_vec[,i],
#                    dev_set, dev_lbls, lambdas)
# }
# 



#Tests
features <- dim(training_set)[2]
theta    <- runif(features, 0, 1)
cost <- logistic_cost(theta, training_set, training_lbls_vec[,1], lambda = 0.1)
grad <- logistic_grad(theta, training_set, training_lbls_vec[,1], lambda = 0.1)

#print(cost)
#print(grad)

#print(dim(as.matrix(cost)))
#print(dim(grad))

#print(dim(as.matrix(training_lbls[,1])))
#res <- train_logistic(training_set, training_lbls[,1], dev_set, dev_lbls, n_lbls)
