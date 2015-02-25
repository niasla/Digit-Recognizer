source("split_data.R")
source("train_logistic.R")
source("logistic_cost.R")
source("logistic_grad.R")
source("normalize_data.R")

# For test purposes, take subset data

n_data <- 20


#Read Test and Training set. Split data also to dev set
train_f <- read.csv("train.csv", colClasses = "numeric",nrows=n_data)
test_f <- read.csv("test.csv", colClasses = "numeric",nrows=n_data)


#dev_prop <- 0.3 



splitted_data <- split_data(train_f,dev_prop=0)
training_set  <- splitted_data[[1]] 
training_lbls <- splitted_data[[2]] 
dev_set       <- splitted_data[[3]]
dev_lbls      <- splitted_data[[4]]
n_lbls        <- 10 

#print(training_lbls)
#Modify labels to become 0's and 1's vector
eye <- diag(n_lbls)
training_lbls <- eye[training_lbls,]
dev_lbls      <- eye[dev_lbls,]

#print(dim(as.matrix(training_lbls)))

#Normalize data 
training_set      <- normalize_data(training_set, epsilon=10^-20)
dev_set           <- normalize_data(dev_set, epsilon=10^-20)
                     

#Expand data by adding a column of 1's

ones_tr  <- matrix(1,dim(training_set)[1],1)
ones_dev <- matrix(1,dim(dev_set)[1],1)

training_set <- cbind(ones_tr,training_set)
dev_set      <- cbind(ones_dev ,dev_set)

## Initializing Logistic Model
features <- dim(training_set)[2]
theta    <- matrix(runif(features, 0, 1), nrow=features)

#Tests

cost <- logistic_cost(theta, training_set, training_lbls[,1], lambda = 0.1)
grad <- logistic_grad(theta, training_set, training_lbls[,1], lambda = 0.1)

#print(cost)
#print(grad)

#print(dim(as.matrix(cost)))
#print(dim(grad))

#print(dim(as.matrix(training_lbls[,1])))
res <- train_logistic(training_set, training_lbls[,1], dev_set, dev_lbls, n_lbls)
