# Load data
load("../myData/Train1.Rdata")
load("../myData/Test1.Rdata")
dataSample <- data.table::fread("../data/sample.csv")

newDataTrain <- as.data.frame(newDataTrain)
newDataTest <- as.data.frame(newDataTest)

# Selection of variables for analysis 
library(tidyverse)
dataTrain <- newDataTrain %>%
  select(-c(ID, DATOP, STD, STA, fechaEspecial))

# caret for partition data
library(caret)
set.seed(123)
indx <- createDataPartition(y = dataTrain$target, times = 1, p = 0.80,
                            list = FALSE)
dfTrain <- dataTrain[indx, ]
dfTest <- dataTrain[-indx, ]

# Categorical features
catFeatures <- names(
  dataTrain %>% select_if(is.factor)
)

# Data for lightgbm
library(lightgbm)
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -1]),
                              label = dfTrain[, 1],
                              categorical_feature = catFeatures)
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -1]),
                             label = dfTest[, 1],
                             categorical_feature = catFeatures)

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.01,
  feature_fraction = 1,
  bagging_fraction = 1,
  #min_data_in_leaf = 100,
  #num_leaves = 255,
  max_depth = -1
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 10000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 500)

#best iter: 1504
#best score: 105.7866

# Predictions
predicciones <- predict(modelo, data.matrix(newDataTest %>%
                                              select(-c(ID, DATOP, STD, STA, fechaEspecial))), 
                        num_iteration = modelo$best_iter)

predicciones[predicciones < 0] <- 0
x11();hist(predicciones)

# Submission
dataSample %>% 
  select(ID) %>% 
  mutate(target = predicciones) ->
  lgbmR9

# Export submission for zindi
write.csv(lgbmR9, file = "../submission/lgbmR9Vuelos.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  slice(1:50) %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()

