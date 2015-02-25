compute_error <- function(data_lbls,classified_lbls){
    err <- mean(data_lbls != classified_lbls)
    err
}