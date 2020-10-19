#  ----- Bibliotecas ------
library(data.table)
library(tidyverse)

# ------- Test ----
load("newTest1.Rdata")

# ------- Train ----
load("newTrain1.Rdata")

# ------- Sample  ----
sampleSub <- fread("../data/sampleSub.csv", encoding = "UTF-8")

# Bibliotecas
library(caret)
library(catboost)

# Datos train
dataTrain <- as.data.frame(newTrain1) %>% 
  relocate(price, everything()) %>% 
  mutate_if(is.character, as.factor)

# Partición de datos (80% train - 20% test)
set.seed(123)
indx <- createDataPartition(y = dataTrain$price, times = 1, p = 0.8, list = FALSE)
dfTrain <- dataTrain[indx, ]
dfTest <- dataTrain[-indx, ]

# Datos para catboost
train_pool <- catboost.load_pool(data = dfTrain[, -1], label = dfTrain[, 1])
test_pool <- catboost.load_pool(data = dfTest[, -1], label = dfTest[, 1])

# Ajuste de modelo
fit_params <- list(iterations = 10000, #default
                   loss_function = 'RMSE',
                   learning_rate = 0.01,
                   train_dir = 'train_dir',
                   logging_level = 'Verbose')
modelo <- catboost.train(train_pool, test_pool, fit_params)


# Predicciones Train
prediccionesTrain <- catboost.predict(modelo, train_pool)
prediccionesTest <- catboost.predict(modelo, test_pool)

# Predicciones Submission
myDataTest2 <- as.data.frame(newTest1) %>%
  mutate_if(is.character, as.factor)
testSubm <- catboost.load_pool(myDataTest2)  
predicciones3 <- catboost.predict(modelo, testSubm)

# Submission
sampleSub %>% 
  select(Id) %>% 
  mutate(price = predicciones3) %>%
  rename(id = Id) %>% 
  as.data.frame() ->
  catBoost3

# Export submission for zindi
write.csv(catBoost3, file = "submission/catBoost3Aptos.csv", row.names = FALSE)

# Puntuación: 0.2735...