#Practical Machine Learning Project

##Background

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, there is data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). The goal of the project is to predict the manner in which they did the exercise.


```r
set.seed(3458965)
```

##Read in the data

```r
training <- read.csv("./pml-training.csv")
testing <- read.csv("./pml-testing.csv")
```

##Load packages

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

##Split data into training (70%) and validation (30%)

```r
intrain <- createDataPartition(y = training$classe, p = 0.7)
intrain <- unlist(intrain)
trainingsubset1 <- training[intrain,]
validation <- training[-intrain,]
```

##Cleaning data
###NAs
Some variables have large amounts of NAs. Checking that gives:

```r
numberofna <- sapply(trainingsubset1, function(x) {sum(is.na(x))})
table(numberofna)
```

```
## numberofna
##     0 13461 
##    93    67
```
As you can see 67 columns are almost entirely NAs

So removing them (and the non-data coloumns):

```r
removedNAcolumns <- names(numberofna[numberofna==13444])
removednondatacolumns <- 1:5
trainingsubset2 <- trainingsubset1[, !names(trainingsubset1) %in% removedNAcolumns]
trainingsubset3 <- trainingsubset2[,-c(1:5)]
```

###Near Zero Variables (NZVs)
Now we'll see if any of the variables are NZVs and if so we shall remove them (to make computation and interpretability easier) and see if the prediction is still valid. If it has an unacceptabley high error ratio we can add these back in and see if it helps

```r
NZV <- nearZeroVar(trainingsubset3, saveMetrics=TRUE)
trainingsubset4 <- trainingsubset3[,!NZV$nzv]
```

##Build a model
As I am not an expert in the field I cannot build a model from hand picked variables. I also cannot tell if a linear model is appropriate and if so under what conditions.
As a result I have used (and found to be the best) a random forest

```r
modelfitRF4 <- train(classe~., data = trainingsubset4, method = "rf")
predictions4 <- predict(modelfitRF4, newdata = validation)
cm <- confusionMatrix(predictions4, validation$classe)
cm
```

Confusion Matrix and Statistics

Overall Statistics
                                          
               Accuracy : 0.9985          
                 95% CI : (0.9971, 0.9993)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9981          

As can be seen this gives an error rate of approximately 0.15% which is incredibly good. Thus it seems adding in the NZVs could do little other than to significantly increase computation time.

##Get the predictions for the test set

```r
predictionstest <- predict(modelfitRF4, newdata = testing)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(predictionstest)
```
