#!/usr/bin/Rscript

source("split_data.R")
source("train_logistic.R")
source("logistic_cost.R")
source("logistic_grad.R")
source("normalize_data.R")
source("classify_logistic.R")
source("compute_error.R")
library("doMC")
library("getopt")

#Defining script options 

spec = matrix(c('training','t',1,"integer",
                'ndata','n',1,"integer",
                'devprop','dp',1,"double",
                'testdevprop','tdp',1,"double",
                'newtonmethod','newton',1,"logical",
                'maxiters','m',1,"integer"),
                byrow=TRUE,ncol=4)

opt = getopt(spec);
# if help was asked for print a friendly message
# and exit with a non-zero error code

#Set some reasonable defaults for the options that are needed,

if ( is.null(opt$training ) ) { opt$training = 0 }
if ( is.null(opt$ndata ) ) { opt$ndata = -1 }
if ( is.null(opt$devprop ) ) { opt$devprop = 0.2 }
if ( is.null(opt$testdevprop ) ) { opt$testdevprop = 0.2 }
if ( is.null(opt$newtonmethod ) ) { opt$newtonmethod = TRUE }
if ( is.null(opt$maxiters ) ) { opt$maxiters = 100 }

dev_prop  = {1-opt$training}*opt$devprop
test_prop = {1-opt$training}*opt$testdevprop

#Rprof("profiling.out")
registerDoMC()

#Read Test and Training set. Split data also to dev set
train_f <- read.csv("train.csv", colClasses = "numeric",nrow = opt$ndata)


splitted_data <- split_data(train_f,dev_prop = dev_prop, test_prop = test_prop)
training_set  <- splitted_data[[1]]
training_lbls <- splitted_data[[2]] 

if (opt$training == 0){
    dev_set       <- splitted_data[[3]]
    dev_lbls      <- splitted_data[[4]]
    test_set  <- splitted_data[[5]]
    test_lbls <- splitted_data[[6]]
}

n_lbls        <- 10 

#Modify labels to become 0's and 1's vector
eye <- diag(n_lbls)
training_lbls_vec <- eye[training_lbls,]


#Normalize training, dev data and test
epsilon <- 10^-10
res_lst      <- normalize_data(training_set, epsilon)
training_set <- res_lst[[1]]
data_mean    <- res_lst[[2]]
data_sd      <- res_lst[[3]]

if(opt$training == 0){
    dev_set  <- normalize_data(dev_set, epsilon,mean=data_mean,s_dev=data_sd)[[1]]
    test_set <- normalize_data(test_set, epsilon,mean=data_mean,
                                   s_dev=data_sd)[[1]]
}

#Expand data by adding a column of 1's

ones_tr  <- matrix(1,dim(training_set)[1],1)
training_set <- cbind(ones_tr,training_set)
if(opt$training == 0){
    ones_dev <- matrix(1,dim(dev_set)[1],1)
    ones_test <- matrix(1,dim(test_set)[1],1)

    dev_set      <- cbind(ones_dev ,dev_set)
    test_set <- cbind(ones_test ,test_set)
    lambdas <-c(0.1,0.5,1,2,3,4,5)
}else
    load("logistic_model.rda")    

#train all logistic models regression model

logistic_models <- foreach(i= 1:n_lbls,.combine=cbind)%dopar%{
    if (opt$training!=0)
        lambdas = model$lambda[i]
    train_logistic(training_set, training_lbls_vec[,i],
                   dev_set, dev_lbls, lambdas, newton=opt$newtonmethod,
                   maxiters=opt$maxiters)
}


# Save Model Parameters
theta <- as.matrix(logistic_models[-1,])
model <- list(data_mean= data_mean,
              data_sd  = data_sd,
              lambda   = logistic_models[1,],
              theta    = theta) 
save(model, file = "logistic_model.rda")

if(opt$training == 0){
    classified_res <- classify_logistic(test_set, theta)
    error_rate <- compute_error(test_lbls,classified_res)
    cat(sprintf("Accuracy: %f%%\n",100*(1-error_rate)))
}else
    cat(sprintf("Done training all the training data.\n"))
#summaryRprof("profiling.out")