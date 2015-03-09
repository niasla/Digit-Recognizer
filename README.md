## Digit Recognizer Competition in Kaggle

Solving Kaggle's Digit Recognizer competition using R language.

#Description

In this I tried to solve this problem using two approaches:
* Logistic Regression.
* Deep Neural Networks.

In both approaches it's important to have in the same directory "train.csv" and "test.csv" files thar is possible to download from Kaggles competition data <a href="https://www.kaggle.com/c/digit-recognizer/data"> here </a>

**Logistic Regression:**
In this solution I used Newton's gradient descent to minimize the cost function with regularization. To run the script first there is a need to train and save the model and afterwards to classify the test data. In the training phase it'll divide the data into training and validation sets to explore the optimal parameter(\lambda). The learning rate \alpha is set to be adaptive thus dividing itself by two when reaching to a local minima. To speed up computations I used the R library *doMC*.

**Deep Neural Networks:**
In this approach I used H2O library for machine learning. 

#Library Dependency
* doMC library.
* H2O  library.