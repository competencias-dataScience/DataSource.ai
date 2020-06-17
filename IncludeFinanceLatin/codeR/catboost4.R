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
  select(-c(newID, uniqueid)) %>% 
  mutate(target = ifelse(target == 1, "Si", "No"))

# Partición de datos (80% train - 20% test)
set.seed(123)
indx <- createDataPartition(y = dataTrain$target, times = 1, p = 0.8, list = FALSE)
dfTrain <- dataTrain[indx, ]
dfTest <- dataTrain[-indx, ]

# Ajuste de modelo
fit_control <- trainControl(
  method = "cv", 
  number = 5,
  search = "random",
  classProbs = TRUE
)
# set grid options
grid <- expand.grid(
  depth = c(6, 8, 15),
  learning_rate = 0.1,
  l2_leaf_reg = c(3, 5),
  rsm = c(1, 0.5, 0.3),
  border_count = c(32, 128),
  iterations = 100
)
model <- caret::train(
  x = dfTrain[, -1], 
  y = dfTrain[, 1],
  method = catboost.caret,
  metric = "Accuracy",
  maximize = TRUE,
  tuneGrid = grid, 
  trControl = fit_control
)

# Best tuning
model$bestTune

# Predicciones
predicciones1 <- predict(model, newdata = dfTest, type = "prob")
predicciones2 <- ifelse(predicciones1$Si > 0.5, "Si", "No")

# Confusion matrix
confusionMatrix(data = factor(predicciones2),
                reference = factor(dfTest$target),
                positive = "Si")

# Accuracy: 0.8835
# Kappa: 0.3915

# Predicciones submission
predicciones3 <- predict(model, newdata = myDataTest, type = "prob")
predicciones4 <- ifelse(predicciones3$Si > 0.5, "Si", "No")

# Submission
dataCatboost4 <- myDataTest %>% 
  mutate(bank_account = predicciones4) %>% 
  select(uniqueid = newID, bank_account) %>% 
  mutate(bank_account = ifelse(bank_account == "Si", "1", "0"))
write.csv(dataCatboost4, file = "submission/catB4.csv", row.names = FALSE)
