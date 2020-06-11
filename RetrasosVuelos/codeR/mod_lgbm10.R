# Mod1 + KFold
# Load data
load("../myData/Train1.Rdata")
load("../myData/Test1.Rdata")
dataSample <- data.table::fread("../data/sample.csv")

newDataTrain <- as.data.frame(newDataTrain)
newDataTest <- as.data.frame(newDataTest)

# Selection of variables for analysis 
library(tidyverse)
dataTrain <- newDataTrain %>%
  select(-c(ID, DATOP, STD, STA))

# Categorical features
catFeatures <- names(
  dataTrain %>% select_if(is.factor)
)

# caret for partition data with resample
library(caret)
set.seed(123)
index <- createDataPartition(y = dataTrain$target, times = 10, p = 0.70,
                             list = TRUE)

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.1,
  feature_fraction = 1,
  bagging_fraction = 1,
  #min_data_in_leaf = 100,
  #num_leaves = 255,
  max_depth = -1
)

# lightgbm with K-Fold manually (k = 10)
library(lightgbm)
k <- 10
predTest <- list()
bestScore <- c()
for (i in 1:k) {
  
  # Data train and test
  dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dataTrain[index[[i]], -1]),
                                label = dataTrain[index[[i]], 1],
                                categorical_feature = catFeatures)
  dataTest_lgbm <- lgb.Dataset(data = data.matrix(dataTrain[-index[[i]], -1]),
                               label = dataTrain[-index[[i]], 1],
                               categorical_feature = catFeatures)
  
  # Train model
  model <- lgb.train(params = myParams,
                     data = dataTrain_lgbm,
                     nrounds = 10000,
                     valids = list(test = dataTest_lgbm),
                     early_stopping_rounds = 500)
  
  # Predictions
  predictions <- predict(model,
                         data.matrix(newDataTest %>%
                                       select(-c(ID, DATOP, STD, STA))),
                         num_iteration = model$best_iter)
  predTest[[i]] = predictions
  
  # Best score for iteration
  bestScore[i] = model$best_score
  
  # Next iteration
  cat("Iteration:==========", i, "RSME:==========", model$best_score, "Ready!")
}

# Mean predictions
dataPred <- as.data.frame(predTest)
names(dataPred) <- paste0("Mod", 1:10)
predicciones <- apply(dataPred, 1, mean)

predicciones[predicciones < 0] <- 0
x11();hist(predicciones)

# Submission
dataSample %>% 
  select(ID) %>% 
  mutate(target = predicciones) ->
  lgbmR10

# Export submission for zindi
write.csv(lgbmR10, file = "../submission/lgbmR10Vuelos.csv", row.names = FALSE)
