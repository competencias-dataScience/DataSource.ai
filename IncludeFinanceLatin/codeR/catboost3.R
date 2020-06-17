# Cargando datos
library(data.table)
library(tidyverse)
load("myDataTrain.Rdata")
load("myDataTest.Rdata")
dataSample <- fread("../data/sample.csv")


# Bibliotecas
library(caret)
library(catboost)

# Datos train
dataTrain <- as.data.frame(myDataTrain) %>% 
  select(-c(newID, uniqueid))

# Partición de datos (80% train - 20% test)
set.seed(123)
indx <- createDataPartition(y = dataTrain$target, times = 1, p = 0.8, list = FALSE)
dfTrain <- dataTrain[indx, ]
dfTest <- dataTrain[-indx, ]

# Datos para catboost
train_pool <- catboost.load_pool(data = dfTrain[, -1], label = dfTrain[, 1])
test_pool <- catboost.load_pool(data = dfTest[, -1], label = dfTest[, 1])

# Ajuste de modelo
fit_params <- list(iterations = 1000,
                   loss_function = 'Logloss',
                   depth = 10,
                   border_count = 128,
                   l2_leaf_reg = 5,
                   learning_rate = 0.01)
modelo <- catboost.train(train_pool, test_pool, fit_params)

# Predicciones
predicciones1 <- catboost.predict(modelo, test_pool,
                                  prediction_type = 'Probability')

predicciones2 <- catboost.predict(modelo, test_pool,
                                  prediction_type = 'Class')

# Confusion matrix
confusionMatrix(data = factor(predicciones2),
                reference = factor(dfTest$target),
                positive = "1")

# Accuracy: 0.8873
# Kappa: 0.4158

# Predicciones submission
myDataTest2 <- myDataTest %>% select(-c(newID, uniqueid))
test_pool2 <- catboost.load_pool(myDataTest2)  
predicciones3 <- catboost.predict(modelo, test_pool2,
                                  prediction_type = 'Class')

# Submission
dataCatboost3 <- myDataTest %>% 
  mutate(bank_account = predicciones3) %>% 
  select(uniqueid = newID, bank_account)
write.csv(dataCatboost3, file = "submission/catB3.csv", row.names = FALSE)
