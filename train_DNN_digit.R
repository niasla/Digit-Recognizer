#!/usr/bin/Rscript

source("split_data.R")
source("normalize_data.R")
source("compute_error.R")
library("doMC")
library("getopt")
library("h2o")

#Run h2o
localH2O <- h2o.init(nthreads=-1,max_mem_size="4096M")


#Defining script options 

spec = matrix(c('nnodes','nn',1,"integer",
                'epochs','ep',1,"integer",
                'ndata','nd',1,"integer",
                'layers','ly',1,"integer"),
               # 'hdroput','hdo',1,"vector"),
                byrow=TRUE,ncol=4)

opt = getopt(spec)
# if help was asked for print a friendly message
# and exit with a non-zero error code

#Set some reasonable defaults for the options that are needed,

if ( is.null(opt$nnodes ) ) { opt$nnodes = 50 }
if ( is.null(opt$epochs ) ) { opt$epochs = 100 }
if ( is.null(opt$layers ) ) { opt$layers = 3 }



dat_h2o <- h2o.importFile(localH2O, path = "train.csv")

if ( is.null(opt$ndata ) ) { opt$ndata = dim(dat_h2o)[1] }

#Split to train & dev sets

sliced_dat_h2o <- dat_h2o[1:opt$ndata,]
train_hex_split <- h2o.splitFrame(sliced_dat_h2o, ratios = 0.8, shuffle = TRUE)

train_h2o <- train_hex_split[[1]]
test_h2o  <- train_hex_split[[2]]


#DNN params
dnn_arch = rep(opt$nnodes, opt$layers)

#Train Deep Model

model <-
    h2o.deeplearning(x = 2:785, # column numbers for predictors
                     y = 1, # column number for label
                     data = train_h2o, # data in H2O format
                     activation = "TanhWithDropout", # or 'Tanh'
                     input_dropout_ratio = 0.2, # % of inputs dropout
                     hidden_dropout_ratios = c(0.5,0.5,0.5),#opt$hdropout, # % for nodes dropout
                     balance_classes = TRUE,
                     hidden = dnn_arch, #c(50,50,50), # three layers of 50 nodes
                     epochs = opt$epochs) # max. no. of epochs 

save(model, file = "DNN_digit_model.rda")
## Using the DNN model for predictions
#print(dim(test_h2o))
#test_h2o[,1]
h2o_yhat_test <- h2o.predict(model, test_h2o[,-1]) 


#print(dim(h2o_yhat_test))
#h2o_yhat_test[,1]
err <- h2o.confusionMatrix(h2o_yhat_test[,1],test_h2o[,1])#compute_error(test_h2o[,1],h2o_yhat_test[,1])


idx = dim(err)[1]

cat("Error Rate: ",err[idx,idx]*100,"%%")