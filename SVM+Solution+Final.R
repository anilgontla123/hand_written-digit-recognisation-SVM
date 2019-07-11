rm(list=ls())
setwd("G:/Upgrad/Course4/Assignment")

############################ SVM Letter Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2  RBF Polynomial
#  4.3 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#The objective is to identify each of the digits in the given data set and classify
#the handwritten digits based on the pixel values given as features.


#####################################################################################
# 2. Data Understanding: 
# Number of Instances: 60,000
# Number of Attributes:785

#Note: #In the given dataset column names are not present and bydefault R provided some random variable names.
#To replace all 785 variable names is very difficult.The column names resembles the pixels details of each and every variable except V1.
#Here V1 is the label which represents the digits(0-9)

#####################################################################################
#3. Data Preparation: 
#####################################################################################

#Load libraries
library(kernlab)
library(readr)
library(caret)
library(ggplot2)
library(gridExtra)
library(dplyr)

#Load the 2 data files "mnist_train" and mnist_test
mnist_train<-read.csv("mnist_train.csv",header = F)
mnist_test<-read.csv("mnist_test.csv",header=F)

# checking for missing values in dataset.Here we will perform all the data cleaning steps only on 
#"mnist_train" data set because this is the data set we have to train and predict the
#results on test data irrespective of how data exists

sapply(mnist_train,function(x) length(which(is.na(x))))  # No NA values
sapply(mnist_train,function(x) length(which(x==" "))) # No blanks

# Lets check for dimensions and its structure
dim(mnist_train)
str(mnist_train)

#Check for summary
summary(mnist_train)

#Here V1 represents digits (0-9).Changing V1 to name "Label"
colnames(mnist_train)[1] <- "Label"
colnames(mnist_test)[1] <- "Label"

# Converting the predicting label to factors
mnist_train$Label<-factor(mnist_train$Label)
mnist_test$Label<-factor(mnist_test$Label)

#Scaling the columns by dividing 255 as the columns are in pixels
mnist_train[, -1] <- mnist_train[, -1]/255
mnist_test[, -1] <- mnist_test[, -1]/255

summary(mnist_train)
summary(mnist_test)

########################################
#EDA Analysis
#######################################

#For understanding the given data, calculated the intensity of pixels for each and every digit.

mnist_train_EDA <-mnist_train
mnist_test_EDA <- mnist_test

mnist_train_EDA$AvgPixelRate <- apply(mnist_train_EDA[,-1],1,FUN = mean)
TotAvgLabelPixelRate<- aggregate(mnist_train_EDA$AvgPixelRate, by=list(mnist_train_EDA$Label),FUN = mean)
mnist_test_EDA$AvgPixelRate <- apply(mnist_test_EDA[,-1],1,FUN = mean)
TotAvgLabelPixelRate<- aggregate(mnist_test_EDA$AvgPixelRate, by=list(mnist_test_EDA$Label),FUN = mean)
ggplot(TotAvgLabelPixelRate,aes(x=Group.1,y=x))+geom_bar(stat="identity")+xlab("Digits")+ylab("Avg Pixel Rate")

#Average intensity is low for digit "1" and for remaing all >0.12

prop.table(table(mnist_test$Label)) # digits 0--->98%,4--->98.2%,8 ---> 97%,6 --->96%,5---> 89%
# remaining all digits are around 10-11% present the given data set

#################################################################
#4. Model Building
#################################################################

set.seed(100)
train.indices = sample(1:nrow(mnist_train), 0.05*nrow(mnist_train))
train = mnist_train[train.indices, ]
test = mnist_test


#4.1  Linear kernel

Model_linear <- ksvm(Label~ ., data = train, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test)
confusionMatrix(Eval_linear,test$Label) 

#Accuracy : 0.9073 
#Kappa : 0.897 
###########################################

#Cross validation for Linear model
############################################
metric <- "Accuracy"  ## Metric <- "Accuracy" implies our Evaluation metric is Accuracy.
#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

trainControl <- trainControl(method="cv", number=5)

set.seed(100)
grid_linear <- expand.grid(.C=seq(1, 5, by=1))

fit.svmLinear <- train(Label~., data=train,method="svmLinear", metric=metric, 
                       tuneGrid=grid_linear, trControl=trainControl)


print(fit.svmLinear)
plot(fit.svmLinear)

##Accuracy :0.8986733 
#Kappa : 0.8873908 
#The final value used for the model was C = 1

#################################################
#4.2 Using RBF Kernel
#################################################

Model_RBF <- ksvm(Label~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test)
confusionMatrix(Eval_RBF,test$Label)   ##confusion matrix - RBF Kernel
##Accuracy : 0.9375 
#Kappa :  0.9305

grid_rbfKernel <- expand.grid(.sigma=c(0.025, 0.05), .C=c(0.1,0.5,1,2) )
fit.rbfKernel <- train(Label~., data=train, method="svmRadial", metric=metric, 
                       tuneGrid=grid_rbfKernel, trControl=trainControl)

print(fit.rbfKernel)


# For sigma = 0.025 and C = 2 
##Accuracy :  0.9503242
#Kappa :  0.9447942

###########################################
# 4.3 RBF polynomial
#############################################

Model_RBFpoly <- ksvm(Label~ ., data = train, scale = FALSE, kernel = "polydot")
Eval_RBFpoly<- predict(Model_RBFpoly, test)
confusionMatrix(Eval_RBFpoly,test$Label)   #confusion matrix - RBFpoly Kernel

##Accuracy :  0.9073
#Kappa : 0.897 

grid_RBFpoly <- expand.grid(.degree = 2,.scale = 0.1,.C=0.1)
fit.RBFPoly <- train(Label~., data=train, method="svmPoly", metric=metric, 
                     tuneGrid=grid_RBFpoly, trControl=trainControl)

print(fit.RBFPoly)

##Accuracy :  0.939331
#Kappa :  0.9325747


#Conclusion: Amnong all the types of kernals, the accuracy is Linear<  polynomial < RBF
#Finally we will consider the model is performig well with type "RBF" having an accuracy 
#in Model building : ##Accuracy : 0.9375       #Kappa :  0.9305

# Also after performing cross validation with 5 folds, the predicted values are as below:
##Accuracy :  0.9503242
#Kappa :  0.9447942             # For sigma = 0.025 and C = 2


################################################################################################
