compute_error <- function(data_lbls,classified_lbls){
    err <- .Internal(mean(data_lbls != classified_lbls))
    err
}