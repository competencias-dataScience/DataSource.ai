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

# Partición de datos (80% train - 20% test)
set.seed(123)
indx <- createDataPartition(y = dataTrain$target, times = 1, p = 0.8, list = FALSE)
dfTrain <- dataTrain[indx, ]
dfTest <- dataTrain[-indx, ]

# Variables categóricas
catFeatures <- names(
 dataTrain %>% select_if(is.factor)
 )

# Datos para lightgbm
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -1]), 
                              label = dfTrain[, 1],
                              categorical_feature = catFeatures)
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -1]),
                             label = dfTest[, 1],
                             categorical_feature = catFeatures)

# Parámetros para lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "binary",
  metric = 'auc',
  learning_rate = 0.01,
  feature_fraction = 1,
  bagging_fraction = 1,
  max_depth = -1,
  is_unbalance = TRUE
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 10000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 50)

# best_iteration: 378
# best_score: 0.8648245
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
predicciones1 <- predict(modelo, data.matrix(myDataTest %>% 
                                               select(-c(newID, uniqueid))))

# Submission
dataLGBM0 <- myDataTest %>% 
  mutate(bank_account = predicciones1) %>% 
  select(uniqueid = newID, bank_account)
write.csv(dataLGBM0, file = "lgbm0.csv", row.names = FALSE)
