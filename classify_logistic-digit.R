#!/usr/bin/Rscript

source("split_data.R")
source("train_logistic.R")
source("logistic_cost.R")
source("logistic_grad.R")
source("normalize_data.R")
source("classify_logistic.R")
source("compute_error.R")

#Load Test File
test_set <- read.csv("test.csv", colClasses = "numeric")

#Split into data and labels
#splitted_data <- split_data(test_f,dev_prop = 0)
#test_set      <- splitted_data[[1]]
#test_lbls     <- splitted_data[[2]]

#Load model
load("logistic_model.rda")
data_mean <- model$data_mean
data_sd   <- model$data_sd
theta     <- model$theta


#Normalize data deoending on training mean and std (from model)
epsilon <- 10^-10
test_set <- normalize_data(test_set, epsilon,mean=data_mean,s_dev=data_sd)[[1]]

#Expand data with  column of 1's

ones_test <- matrix(1,dim(test_set)[1],1)
test_set <-  cbind(ones_test,test_set)

#Classify test set
classified_res <- classify_logistic(test_set, theta)
result <- cbind(ImageId={1:dim(test_set)[1]},Label={classified_res-1})
#colnames(classified_res) <- c("ImageId","Label")

#Write classification to a csv file
write.table(result, file="digit_recognizer_logistic.csv", sep=",",row.names=FALSE)
