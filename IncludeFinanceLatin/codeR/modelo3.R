# Cargando datos
library(data.table)
library(tidyverse)
load("myDataTrain.Rdata")
load("myDataTest.Rdata")
dataSample <- fread("../data/sample.csv")


# Bibliotecas
library(caret)
library(lightgbm)

# Datos train
dataTrain <- as.data.frame(myDataTrain) %>% 
  select(-c(newID, uniqueid))

# One-Hote Encoding (previous)
library(recipes)
rec <- recipe(target ~ ., data = dataTrain)
dummies <- rec %>% 
  step_dummy(all_nominal(), one_hot = TRUE)
dummies <- prep(dummies, training = dataTrain)
dataTrain2 <- bake(dummies, new_data = dataTrain) %>% 
  select(target, everything()) %>% 
  as.data.frame()

# Datos test
recTest <- recipe(~., data = myDataTest %>% select(-c(newID, uniqueid)))
dumTest <- recTest %>% 
  step_dummy(all_nominal(), one_hot = TRUE)
dumTest <- prep(dumTest, training = myDataTest)
dataTest2 <- bake(dumTest, new_data = myDataTest) %>%  
  as.data.frame()

# Partición de datos (80% train - 20% test)
set.seed(123)
indx <- createDataPartition(y = dataTrain2$target, times = 1, p = 0.8, list = FALSE)
dfTrain <- dataTrain2[indx, ]
dfTest <- dataTrain2[-indx, ]

# Variables categóricas
#catFeatures <- names(
#  dataTrain %>% select_if(is.factor)
# categorical_feature = catFeatures)

# Datos para lightgbm
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -1]), 
                              label = dfTrain[, 1])
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -1]),
                             label = dfTest[, 1])

# Parámetros para lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "binary",
  metric = 'auc',
  learning_rate = 0.001,
  feature_fraction = 1,
  bagging_fraction = 1,
  max_depth = -1,
  is_unbalance = TRUE,
  min_data_in_leaf = 100
  #num_leaves = 64,
  #subsample_for_bin = 200,
  #reg_alpha = 1.2,
  #reg_lambda = 1.2,
  #min_split_gain = 0.5,
  #min_child_weight = 1,
  #min_child_samples = 5,
  #scale_pos_weight = 1,
  #num_class = 1
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 10000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 50)

# best_iteration: 1757
# best_score: 0.8647503
modelo$best_iter
modelo$best_score

# --------- Predichos test (train)
predicciones0 <- predict(modelo, data.matrix(dfTest %>% 
                                               select(-target)))

# Confusion matrix
cut0.5 <- factor(ifelse(predicciones0 > 0.5, "1", "0"))
confusionMatrix(data = cut0.5,
                reference = factor(dfTest$target),
                positive = "1")

# --------- Predichos test (submission)

# Predicciones
predicciones1 <- predict(modelo, data.matrix(dataTest2))


# Submission
dataLGBM3 <- myDataTest %>% 
  mutate(bank_account = if_else(predicciones1 > 0.5, "1", "0")) %>% 
  select(uniqueid = newID, bank_account)
write.csv(dataLGBM3, file = "submission/lgbm3.csv", row.names = FALSE)
