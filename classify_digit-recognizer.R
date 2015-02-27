source("split_data.R")
source("train_logistic.R")
source("logistic_cost.R")
source("logistic_grad.R")
source("normalize_data.R")
source("classify_logistic.R")
source("compute_error.R")

#Load Test File
test_f <- read.csv("test.csv", colClasses = "numeric", nrow = n_data)

#Split into data and labels
splitted_data <- split_data(test_f,dev_prop = 0)
test_set      <- splitted_data[[1]]
test_lbls     <- splitted_data[[2]]

#Load model


#Normalize data deoending on training mean and std (from model)