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

spec = matrix(c('architecture','arch',1,"character",
                'epochs','ep',1,"integer",
                'ndata','nd',1,"integer",
                'indropout','ido' ,1,"integer",
                'nsegments','nseg' ,1,"integer",
                'learningcurves','lc',0,"logical",
                'classify','cls',0,"logical",
                'hdropout' ,'hdo',1,"character"),
                byrow=TRUE,ncol=4)

opt = getopt(spec)
# if help was asked for print a friendly message
# and exit with a non-zero error code

#Set some reasonable defaults for the options that are needed,

#DNN params
if ( is.null(opt$architecture) ){
    dnn_arch <- c(50,50,50)  
}else{
    dnn_arch <- as.integer(strsplit(opt$architecture," ")[[1]])   
}  


if ( is.null(opt$epochs ) ) { opt$epochs <- 100 }
if ( is.null(opt$nsegments ) ) { opt$nsegments <- 20 }
if ( is.null(opt$indropout ) ) { opt$indropout <- 0.2 }

if ( is.null(opt$learningcurves ) ) { opt$learningcurves <- F 
}else opt$learningcurves=T

if ( is.null(opt$classify ) ) { opt$classify <- F 
}else opt$classify=T

if ( is.null(opt$hdropout) ){ 
    hidden_dropout <- rep(0.5,length(dnn_arch))  
}else{
    hidden_dropout <- as.integer(strsplit(opt$architecture," ")[[1]])   
}


if (opt$classify){
    dl_model <- h2o.loadModel(localH2O, path = "/home/nizar/kaggle/Digit-Recognizer/h2o_DNN_digit_model")
    dat_h2o <- h2o.importFile(localH2O, path = "test.csv")
    h2o_yhat_test <- h2o.predict(dl_model[[1]], dat_h2o)
#    labels <- as.matrix.H2OParsedData(h2o_yhat_test[1,])
    
    
 #   result <- cbind(ImageId={1:dim(labels)[1]},Label={classified_res-1})
 #   write.table(result, file="digit_recognizer_DNN.csv", sep=",",row.names=FALSE)
    
    
}else{

    
    
    dat_h2o <- h2o.importFile(localH2O, path = "train.csv")
    
    if ( is.null(opt$ndata ) ) { opt$ndata = dim(dat_h2o)[1] }
    
    
    
    if(opt$learningcurves){
        segs <- opt$nsegments
        rate <- opt$ndata/segs    
        data_segments <- seq(from=rate,to=opt$ndata,by=rate)
        
        train_err <- vector(length=segs)
        valid_err <- vector(length=segs)
        idx <- 1
        for (data_limit in data_segments){
            
            #Split to train & dev sets
            sliced_dat_h2o <- dat_h2o[1:data_limit,]
            train_hex_split <- h2o.splitFrame(sliced_dat_h2o, 
                                              ratios = 0.8,
                                              shuffle = TRUE)
            
            train_h2o <- train_hex_split[[1]]
            valid_h2o <- train_hex_split[[2]]
            #test_h2o  <- train_hex_split[[3]]
            
            
            dl_model <-
                h2o.deeplearning(x = 2:785, # column numbers for predictors
                                 y = 1, # column number for label
                                 data = train_h2o, # data in H2O format
                                 validation = valid_h2o,
                                 activation = "RectifierWithDropout", # or 'Tanh'
                                 input_dropout_ratio = opt$indropout, # % of inputs dropout
                                 hidden_dropout_ratios = hidden_dropout,#opt$hdropout, # % for nodes dropout
                                 balance_classes = TRUE,
                                 hidden = dnn_arch, #c(50,50,50), # three layers of 50 nodes
                                 epochs = opt$epochs) # max. no. of epochs 
            
           train_err[idx] <- dl_model@model$train_class_err*100
           valid_err[idx] <- dl_model@model$valid_class_err*100
           idx <- idx+1
            
        }
        
        #pdf(width = 1024,height=1024)
        y_range <- range(0,train_err,valid_err)
        plot(train_err,type="o",col="red",ylim=y_range,axes = F,ann = F)
        
        axis(1 , at=1:length(data_segments), labels=data_segments)
        
        y_ticks <- seq(from=0,to=100,by=10)
        axis(2 ,las=1, at=y_ticks)
        box()
        
        lines(valid_err,type="o",col="blue")
        
        title(xlab="Number of Data", col.lab=rgb(0,0,0))
        title(ylab="Error Rate(%)", col.lab=rgb(0,0,0))
        
        legend(1, y_range[2], c("Training Error","Validation Error"),col=c("red","blue")
               ,cex=0.8, pch=21:22, lty=1:2)
        
        
    }else{    
        #Split to train & dev sets
        sliced_dat_h2o <- dat_h2o[1:opt$ndata,]
        train_hex_split <- h2o.splitFrame(sliced_dat_h2o, ratios = 0.8, shuffle = TRUE)
        
        train_h2o <- train_hex_split[[1]]
        valid_h2o <- train_hex_split[[2]]
        
        
    #Train Deep Model
    
    dl_model <-
        h2o.deeplearning(x = 2:785, # column numbers for predictors
                         y = 1, # column number for label
                         data = train_h2o, # data in H2O format
                         validation = valid_h2o,
                         activation = "RectifierWithDropout", # or 'Tanh'
                         input_dropout_ratio = opt$indropout, # % of inputs dropout
                         hidden_dropout_ratios = hidden_dropout,#opt$hdropout, # % for nodes dropout
                         balance_classes = TRUE,
                         hidden = dnn_arch, #c(50,50,50), # three layers of 50 nodes
                         epochs = opt$epochs) # max. no. of epochs 
    
        h2o.saveModel(dl_model, dir="/home/nizar/kaggle/Digit-Recognizer/",name ="h2o_DNN_digit_model", force=T)
        
        #dl_model <- h2o.loadModel(localH2O, path = "/home/nizar/kaggle/Digit-Recognizer/h2o_DNN_digit_model")
        ## Using the DNN model for predictions
        #print(dim(test_h2o))
        #test_h2o[,1]
        #h2o_yhat_test <- h2o.predict(model, test_h2o[,-1]) 
        
        
        #print(dim(h2o_yhat_test))
        #h2o_yhat_test[,1]
        #err <- h2o.confusionMatrix(h2o_yhat_test[,1],test_h2o[,1])#compute_error(test_h2o[,1],h2o_yhat_test[,1])
        #idx = dim(err)[1]
        
        #model
        cat("Training Error Rate: ",dl_model@model$train_class_err*100,"%\n")
        cat("Validation Error Rate: ",dl_model@model$valid_class_err*100,"%\n")
    
        test_dat_h2o <- h2o.importFile(localH2O, path = "test.csv")
        h2o_yhat_test <- h2o.predict(dl_model, test_dat_h2o)
        
        
        labels <- as.matrix.H2OParsedData(h2o_yhat_test[,1])
        result <- cbind(ImageId={1:dim(labels)[1]},Label={labels[,1]})
        write.table(result, file="digit_recognizer_DNN.csv", sep=",",row.names=FALSE)
    
        #cat("Test Error Rate: ",err[idx,idx]*100,"%\n")
       
    }
}