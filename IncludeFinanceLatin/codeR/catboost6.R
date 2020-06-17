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

# Datos para catboost
train_pool <- catboost.load_pool(data = dfTrain[, -1], label = dfTrain[, 1])
test_pool <- catboost.load_pool(data = dfTest[, -1], label = dfTest[, 1])

# Ajuste de modelo
fit_params <- list(iterations = 5000,
                   loss_function = 'Logloss',
                   depth = 6,
                   border_count = 32,
                   rsm = 0.5,
                   l2_leaf_reg = 3,
                   learning_rate = 0.01,
                   od_type = 'Iter',
                   use_best_model = TRUE,
                   od_wait = 500)
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
# Kappa: 0.4186

# Predicciones submission
test_pool2 <- catboost.load_pool(dataTest2)  
predicciones3 <- catboost.predict(modelo, test_pool2,
                                  prediction_type = 'Class')

# Submission
dataCatboost6 <- myDataTest %>% 
  mutate(bank_account = predicciones3) %>% 
  select(uniqueid = newID, bank_account)
write.csv(dataCatboost6, file = "submission/catB6.csv", row.names = FALSE)
