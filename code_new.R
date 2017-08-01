-------------------------------------------------------------------------------
# Ultimate-Student-Hunt-Code

-------------------------------------------------------------------------------
# Library packages used

library(gbm) # gbm model # version 2.1.1
library(caret) # for dummyVars # version 6.0-71
library(Metrics) # calculate RMSE and MAE errors # version 0.1.1
library(xgboost) # xgboost model # version 0.4-4
library(plyr) # Tools for Splitting, Applying and Combining Data # Version 6.0-71
library(doParallel) # provides drop- in replacements for most of the functionality of those packages, with integrated handling of random-number generation # version 3.2.3
library(foreach) # provides a new looping construct for executing R code repeatedly  # version 1.4.3
library(doSNOW) # provides drop- in replacements for most of the functionality of those packages, with integrated handling of random-number generation # version 3.2.3
library(dummies) # create dummy variables # version 1.5.6
library(caTools) # for splitting data # version 1.17.1
library(neuralnet) # running neural network model # version 1.33
library(nnet) # running neural network model with cv # version 7.3-11
library(date) # splitting date into month, day and year variable # version 1.2-34
library(mice) # predicting and imputing missing data # version 2.25
library(caTools) #R 3.3.1 version
library(lubridate)#Date to day,month & year R 3.3.1 version


---------------------------------------------------------------------------------
  
setwd("F:/DataScience/av_student_hunt/data/Train")
----------------------------------------------------------------------------
# Run the test data and read it 

Test <- read.csv("F:/DataScience/av_student_hunt/data/Test.csv")

# View the test data

View(Test)
----------------------------------------------------------------------------
# Run the train data and read it

Train <- read.csv("F:/DataScience/av_student_hunt/data/Train.csv")

# View the train data

View(Train)
----------------------------------------------------------------------------
  
# For Train data--> generating new varible 

Train$Date_new <- paste0(substr(Train$Date, 7, 10),"-",substr(Train$Date, 4, 5),"-",substr(Train$Date, 1, 2)) 

#Converting date to Day,Month and year in Train set

Train$Day <- mday(Train$Date_new)
Train$Year <- year(Train$Date_new)
Train$Month <- month(Train$Date_new)

------------------------------------------------------------------------------
  
#Removing variables ID,ParkID,Location,Date and Date_new 

Train$ID<- NULL
Train$Park_ID<- NULL
Train$Date<- NULL
Train$Date_new<- NULL
Train$Location_Type<- NULL

--------------------------------------------------------------------------------
  
#Replacing NA values with -999

Train[is.na(Train)]<--999
----------------------------------------------------------------------------------

# For Test  data --> generating new variable

Test$Date_new <- paste0(substr(Test$Date, 7, 10),"-",substr(Test$Date, 4, 5),"-",substr(Test$Date, 1, 2)) 

#Converting date to Day,Month and year in Train set
Test$Day <- mday(Test$Date_new)
Test$Year <- year(Test$Date_new)
Test$Month <- month(Test$Date_new)
--------------------------------------------------------------------------------
  
#Removing variables ID,ParkID,Location,Date and Date_new 

Test$ID<- NULL
Test$Park_ID<- NULL
Test$Date<- NULL
Test$Date_new<- NULL 
Test$Location_Type<- NULL
---------------------------------------------------------------------------------
  
#Replacing NA values with -999

Test[is.na(Train)]<--999
---------------------------------------------------------------------------------
# GBM MODEL with rearranging data
# replacing missing values by -999 and converting date to day, month and year and removed the variables Location, Park ID, ID

library(gbm)
Model.gbm = gbm(Footfall ~.,distribution="laplace", data=Train, n.trees=5000, interaction.depth =6, shrinkage=0.05, n.minobsinnode = 10)
# outputfile 
Prediction_gbm = predict.gbm(Model.gbm, Test, n.trees=5000)
mysolution = data.frame( ID = Test$ID, Footfall = Prediction_gbm)
write.csv(mysolution, file = "gbm.csv", row.names = FALSE)

importance = summary.gbm(Model.gbm, plotit=TRUE)
-------------------------------------------------------------------------------------

# XG-BOOST MODEL.... REARRANGING THE DATA AGAIN 

# again running train and test data

-------------------------------------------------------------------------

train <- read.csv('file:///F:/DataScience/av_student_hunt/data/Train.csv')
test <- read.csv('file:///F:/DataScience/av_student_hunt/data/Test.csv')

--------------------------------------------------------------------------------------

temp_train = train
temp_train$Footfall = NULL  # to remove footfall variable from train data

total = rbind(temp_train, test)
rm(temp_train)

---------------------------------------------------------------------------------------------
# feature engineering
total$diff_breeze_speed = total$Max_Breeze_Speed-total$Min_Breeze_Speed
total$diff_Atmospheric_Pressure = total$Max_Atmospheric_Pressure-total$Min_Atmospheric_Pressure
total$diff_Ambient_Pollution = total$Max_Ambient_Pollution-total$Min_Ambient_Pollution
total$diff_moisture = total$Max_Moisture_In_Park-total$Min_Moisture_In_Park
total$Date_new <- paste0(substr(total$Date, 7, 10),"-",substr(total$Date, 4, 5),"-",substr(total$Date, 1, 2)) 
total$Month <- month(total$Date_new)

total$Date_new = NULL

-------------------------------------------------------------------------------------------------
# some more feature engineering added later
total$avg_breeze_speed =          (total$Max_Breeze_Speed+total$Min_Breeze_Speed)/2
total$avg_Atmospheric_Pressure =  (total$Max_Atmospheric_Pressure+total$Min_Atmospheric_Pressure)/2
total$avg_Ambient_Pollution =     (total$Max_Ambient_Pollution+total$Min_Ambient_Pollution)/2
total$avg_moisture =              (total$Max_Moisture_In_Park+total$Min_Moisture_In_Park)/2

-----------------------------------------------------------------------------------------------
# feature engineering for direction of wind 
summary(total$Direction_Of_Wind)
total$wind_dir_categorical[is.na(total$Direction_Of_Wind )] <- 5 #assigning 1 to missing values 
total$wind_dir_categorical[total$Direction_Of_Wind<=90] <- 1 
total$wind_dir_categorical[total$Direction_Of_Wind >90 & total$Direction_Of_Wind <=180 ] <- 2
total$wind_dir_categorical[total$Direction_Of_Wind >180 & total$Direction_Of_Wind <=270 ] <- 3 
total$wind_dir_categorical[total$Direction_Of_Wind >270 & total$Direction_Of_Wind <=360 ]  <- 4
summary(total$wind_dir_categorical)

names(total)

-----------------------------------------------------------------------------------------------
  
# removing only id and date 
remove_column=c("ID","Date")

total=total[,!(names(total) %in% remove_column)]
names(total)

# next line creates a data frame with additional features of dummy variables using specified 4 features
library(dummies)
total_with_dummy <- dummy.data.frame(total, names=c("month","Park_ID","wind_dir_categorical","Location_Type"), sep="_")

total_with_dummy <- data.frame(total_with_dummy)
---------------------------------------------------------------------------------------

# spliting the data in original format 
training_with_dummy = total_with_dummy[1:nrow(train),]
testing_with_dummy = total_with_dummy[114540:nrow(total),]
rm(total_with_dummy)
----------------------------------------------------------------------------------------
  
x <- train$Footfall
# now creating the model 
# creating first matrix (data and label) using train
dtrain<-xgb.DMatrix(data=data.matrix(training_with_dummy),label=data.matrix(x),missing=NA)
watchlist<-list(train=dtrain)


param <- list(  objective           = "reg:linear", 
                booster             = "gbtree",
                eta                 = 0.01,
                max_depth           = 10 ,
                subsample           = 0.6,
                colsample_bytree    = 0.9
)

set.seed(10000)
# creating the model 
clf <- xgb.train(   params              = param, 
                    data                = dtrain,
                    nrounds             = 1000, 
                    verbose             = 2,
                    #early.stop.round    = 150,
                    watchlist           = watchlist,
                    maximize            = TRUE,
                    eval_metric       = "rmse",
                    print.every.n = 50
)

pred_exp=predict(clf,data.matrix(testing_with_dummy),missing=NA)

#saving the raw output
z <- data.frame(pred_exp)

---------------------------------------------------------------------------------------------
# neural network

set.seed(1234567890)
install.packages("neuralnet")
library(neuralnet)
train <- read.csv('file:///F:/Data Science/av_student_hunt/data/Train.csv')
train$Avg_Ambient_Pollution <- (train$Max_Ambient_Pollution+train$Min_Ambient_Pollution)/2
train$Date_new <- paste0(substr(train$Date, 7, 10),"-",substr(train$Date, 4, 5),"-",substr(train$Date, 1, 2)) 
train$month <- month(train$Date_new)
------------------------------------------------------------------------------------------
  
# removing the rows which have missing values in Average_Atmospheric_Pressure and Avg_Ambient_Pollution 
train_deleted <- train[!(is.na(train$Average_Atmospheric_Pressure)) | !(is.na(train$Avg_Ambient_Pollution)),]

-------------------------------------------------------------------------------------------
  
library(mice)
summary(train_deleted)
simple_nn = train_deleted[c("Location_Type", "Park_ID", "Direction_Of_Wind", "month" , "Average_Breeze_Speed" , "Var1" , 
                            "Avg_Ambient_Pollution", "Average_Moisture_In_Park" , "Average_Atmospheric_Pressure")]
summary(simple_nn)
set.seed(144)
imputed_nn = complete(mice(simple_nn))
-------------------------------------------------------------------------------------------
  
# now replacing columns having NA values with imputed columns
summary(imputed_nn)
train_deleted$Direction_Of_Wind = imputed_nn$Direction_Of_Wind
train_deleted$Average_Breeze_Speed = imputed_nn$Average_Breeze_Speed
train_deleted$Var1 = imputed_nn$Var1
train_deleted$Avg_Ambient_Pollution = imputed_nn$Avg_Ambient_Pollution
train_deleted$Average_Moisture_In_Park = imputed_nn$Average_Moisture_In_Park
train_deleted$Average_Atmospheric_Pressure = imputed_nn$Average_Atmospheric_Pressure
train_deleted$Var2 = log(train_deleted$Var1)
train_deleted$Var2[train_deleted$Var2 == -Inf] = 0

train_deleted$month = as.numeric(as.character(train_deleted$month))
--------------------------------------------------------------------------------------------
  

# spliting into training and testing data set

library(caTools)
set.seed(3000)
spl = sample.split(train_deleted$Footfall, SplitRatio = 0.7)
Train_nn = subset(train_deleted, spl == TRUE)
Test_nn = subset(train_deleted, spl == FALSE)
------------------------------------------------------------------------------------------------
#Preparing the model
neural = neuralnet(Footfall ~ month + Direction_Of_Wind + Average_Breeze_Speed + Var2 + Avg_Ambient_Pollution + Average_Moisture_In_Park + Average_Atmospheric_Pressure +Location_Type, Train_nn, hidden=4, linear.output = TRUE, threshold = 0.1, lifesign = "minimal")
-------------------------------------------------------------------------------------------------
  
# now preparing the test data for prediction 
test <- read.csv('file:///F:/Data Science/av_student_hunt/data/Test.csv')
test$Avg_Ambient_Pollution <- (test$Max_ambient_pollution+test$Min_ambient_pollution)/2
test$Date_new <- paste0(substr(test$Date, 7, 10),"-",substr(test$Date, 4, 5),"-",substr(test$Date, 1, 2)) 
test$month <- month(test$Date_new)

test$Var2 = log(test$Var1)
test$Var2[test$Var2 == -Inf] = 0

test_final <- subset(test, select = c("month" , "Direction_Of_Wind" , "Average_Breeze_Speed" , "Var2" , 
                                      "Avg_Ambient_Pollution" , "Average_Moisture_In_Park" , "Average_Atmospheric_Pressure" , "Location_Type"))
-------------------------------------------------------------------------------------------------
  
# now making the prediction on actual data
prediction_nn <- compute(neural,test_final)

mysolution = data.frame(ID = test$ID, Footfall = prediction_nn)

--------------------------------------------------------------------------------------

xgb = z
neural = mysolution

# combining neural_net and xgb of 115.85
submission_combo = data.frame( ID = output$ID, Footfall =((xgb$Footfall)*0.6667+(neural$Footfall)*0.3333))
                               
combined_net_xgb_other_114.13 = submission_combo 
-----------------------------------------------------------------------------------------------
# combining gbm+XGboost+ANN model
# providing appropriate weightage

pred_ens = (2.5*gbm$Footfall + 7.5*combined_net_xgb_other_114.13$Footfall)/10
mysolution = data.frame( ID = Test$ID, Footfall = pred_ens)
-----------------------------------------------------------------------------------------------
# write the final prediction
write.csv(mysolution, file = "final_prediction.csv", row.names = FALSE)

#-----------------------------------------------------------------------------------------------

                               
                             
