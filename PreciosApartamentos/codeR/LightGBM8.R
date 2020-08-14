#  ----- Bibliotecas ------
library(lightgbm)
library(data.table)
library(tidyverse)
library(fastDummies)

# ------- Test ----
test <- fread("../data/test.csv", encoding = "UTF-8") %>% 
  select(-c(Id, property_type, operation_type, currency)) %>% 
  mutate(rooms = factor(rooms),
         bedrooms = factor(bedrooms),
         bathrooms = factor(bathrooms)) %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate(surfaceClass = if_else(surface_total <= 55, true = "Pequeña",
                                false = if_else(
                                  surface_total > 55 & surface_total <= 105,
                                  true = "Mediana",
                                  false = "Grande"
                                ))) %>% 
  as.data.frame() 

test_dummy <- dummy_cols(test)

# ------- Train ----
train <- fread("../data/train.csv", encoding = "UTF-8") %>% 
  select(-c(Id, property_type, operation_type, currency)) %>% 
  mutate(rooms = factor(rooms),
         bedrooms = factor(bedrooms),
         bathrooms = factor(bathrooms)) %>% 
  mutate_if(is.character, as.factor) %>% 
  relocate(price) %>% 
  mutate(surfaceClass = if_else(surface_total <= 55, true = "Pequeña",
                                false = if_else(
                                  surface_total > 55 & surface_total <= 105,
                                  true = "Mediana",
                                  false = "Grande"
                                ))) %>% 
  as.data.frame()

train_dummy <- dummy_cols(train)

train_dummy2 <- train_dummy[, names(train_dummy) %in% names(test_dummy)]
train_dummy2$price <- train$price

test_dummy2 <-  test_dummy[, names(test_dummy) %in% names(train_dummy2)]

# ------- Sample  ----
sampleSub <- fread("../data/sampleSub.csv", encoding = "UTF-8")

# caret for partition data
library(caret)
set.seed(123)
indx <- createDataPartition(y = train_dummy2$price, times = 1, p = 0.80,
                            list = FALSE)
dfTrain <- train_dummy2[indx, ]
dfTest <- train_dummy2[-indx, ]

# Data for lightgbm
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -241]),
                              label = dfTrain[, 241])
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -241]),
                             label = dfTest[, 241])

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "poisson",
  metric = "rmse",
  learning_rate = 0.01,
  feature_fraction = 0.3,
  bagging_fraction = 0.7,
  max_depth = -1
  #min_data_in_leaf = 100,
  #num_leaves = 500,
  
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 10000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 500)

# Best iter:8102
# Best score: 70264.64

# Predictions
predicciones <- predict(modelo, data.matrix(test_dummy2), 
                        num_iteration = modelo$best_iter)

predicciones[predicciones < 0] <- 0
x11();hist(predicciones)

# Submission
sampleSub %>% 
  select(Id) %>% 
  mutate(price = predicciones) %>% 
  as.data.frame() ->
  lgbmR8

# Export submission for zindi
write.csv(lgbmR8, file = "submission/lgbmR8Aptos.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  slice(1:50) %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()