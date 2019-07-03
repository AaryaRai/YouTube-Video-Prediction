library(lattice)
library(ggplot2)
library(caret)
library(FNN)
library(MASS)
library(gains)
library(rpart)
library(rpart.plot)
library(ISLR)
library(class)

all_videos <- na.omit(read.csv("Allvideos.csv"))

### Partitioning the data into training and validation data 

set.seed(123)

training.index <- createDataPartition(all_videos$category_id, p = 0.8, list = FALSE)

data.train <- all_videos[training.index, ]
data.valid <- all_videos[-training.index, ]

data.train.x = data.train[,c("views","likes","dislikes","comment_count")]
data.train.y = data.train[,c("category_id")]

data.valid.x = data.valid[,c("views","likes","dislikes","comment_count")]
data.valid.y = data.valid[,c("category_id")]

norm.values  <- preProcess(data.train.x, method = c("center", "scale"))

data.train.norm <- predict(norm.values, data.train.x)
data.valid.norm <- predict(norm.values, data.valid.x)

data.train.norm = cbind(data.train.norm,category_id = data.train.y)
data.valid.norm = cbind(data.valid.norm,category_id = data.valid.y)

## Predicting the accuracy with the LDA model
lda1 <- lda(category_id~views+likes+dislikes+comment_count, data = data.train.norm)
lda1

pred <- predict(lda1, data.valid.norm)
names(pred)

#Checking the model accuracy

print("Confusion Matrix")
# Predicted v/s Actual accuracy
print(table(pred$class, data.valid.norm$category_id))  # pred v actual

# Percent accuracy of model
print("Accuracy")
print(mean(pred$class == data.valid.norm$category_id))  # percent accurate



##Predicting the accuracy with the CLASSIFICATION TREES model
default.ct <- rpart(category_id ~ views+likes+dislikes+comment_count, data = data.train.norm, method = "class")
prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 2, varlen = -10)

### Complexity Parameters
### xval:  Number of folds to use cross-validation procedure
### CP: sets the smallest value for the complexity paraeter
cv.ct <- rpart(category_id ~ views+likes+dislikes+comment_count, data = data.train.norm, method = "class", 
               cp = 0.00001, minsplit = 5, xval = 5)
printcp(cv.ct)

### Obtaining the Best-Pruned Tree
set.seed(1)
cv.ct <- rpart(category_id ~ views+likes+dislikes+comment_count, data = data.train.norm, method = "class", cp = 0.00001, minsplit = 1, xval = 5)  # minsplit is the minimum number of observations in a node for a split to be attempted. xval is number K of folds in a K-fold cross-validation.
printcp(cv.ct)  # Print out the cp table of cross-validation errors. The R-squared for a regression tree is 1 minus rel error. xerror (or relative cross-validation error where "x" stands for "cross") is a scaled version of overall average of the 5 out-of-sample errors across the 5 folds.
pruned.ct <- prune(cv.ct, cp = 0.0154639)
prp(pruned.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white')) 

#Accuracy for the Pruned Tree

### Confusion Matrices

### for Training set
pruned.ct.point.pred.train <- predict(pruned.ct, 
                                       data = data.train.norm, 
                                       type = "class")
confusionMatrix(pruned.ct.point.pred.train, as.factor(data.train.norm$category_id))

### for Validation set
pruned.ct.point.pred.valid <- predict(pruned.ct, 
                                       newdata = data.valid.norm, 
                                       type = "class")
confusionMatrix(pruned.ct.point.pred.valid, as.factor(data.valid.norm$category_id))


##Predicting Accuracy with the KNN MODEL

nn <- knn(train = data.train.norm[, c("views","likes","dislikes","comment_count")], test = data.valid.norm[,c("views","likes","dislikes","comment_count")], cl = data.train.norm[, c("category_id")], k = 5)


#Get the confusion matrix to see accuracy value and other parameter values

data.valid.norm$category_id = as.factor(data.valid.norm$category_id)
confusionMatrix(nn, data.valid.norm$category_id)

##So concluding from all the above models we can say that KNN model predicts the best
##in this case with the accuracy of 85.39 %.