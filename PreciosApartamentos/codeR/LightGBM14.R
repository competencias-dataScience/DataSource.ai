#  ----- Bibliotecas ------
library(lightgbm)
library(data.table)
library(tidyverse)
library(fastDummies)

# ------- Test ----
load("newTest1.Rdata")
test <- newTest1 %>% 
  mutate(rooms = factor(rooms),
         bedrooms = factor(bedrooms),
         bathrooms = factor(bathrooms)) %>% 
  mutate_if(is.character, as.factor) %>% 
  as.data.frame() 

test_dummy <- dummy_cols(test) %>% 
  select(-c(rooms, bedrooms, bathrooms, pais, provincia_departamento,
            ciudad, Capital, surfaceClass))

# ------- Train ----
load("newTrain1.Rdata")
train <- newTrain1 %>% 
  mutate(rooms = factor(rooms),
         bedrooms = factor(bedrooms),
         bathrooms = factor(bathrooms)) %>% 
  mutate_if(is.character, as.factor) %>% 
  relocate(price) %>% 
  as.data.frame()

train_dummy <- dummy_cols(train) %>% 
  select(-c(rooms, bedrooms, bathrooms, pais, provincia_departamento,
            ciudad, Capital, surfaceClass))

train_dummy2 <- train_dummy[, names(train_dummy) %in% names(test_dummy)]
train_dummy2$price <- train$price

test_dummy2 <-  test_dummy[, names(test_dummy) %in% names(train_dummy2)]

# ------- Sample  ----
sampleSub <- fread("../data/sampleSub.csv", encoding = "UTF-8")

# caret for partition data
library(caret)
set.seed(123)
indx <- createDataPartition(y = train_dummy2$price, times = 1, p = 0.90,
                            list = FALSE)
dfTrain <- train_dummy2[indx, ]
dfTest <- train_dummy2[-indx, ]

# Data for lightgbm
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -238]),
                              label = dfTrain[, 238])
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -238]),
                             label = dfTest[, 238])

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "poisson",
  metric = "rmse",
  learning_rate = 0.01,
  feature_fraction = 0.7,
  bagging_fraction = 1
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 20000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 500)

# Best iter:4889
# Best score: 66752.71

# Predictions
predicciones <- predict(modelo, data.matrix(test_dummy2), 
                        num_iteration = modelo$best_iter)

predicciones[predicciones < 0] <- 0
x11();hist(predicciones)

# Submission
sampleSub %>% 
  select(Id) %>% 
  mutate(price = predicciones) %>%
  rename(id = Id) %>% 
  as.data.frame() ->
  lgbmR14

# Export submission for zindi
write.csv(lgbmR14, file = "submission/lgbmR14Aptos.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  slice(1:50) %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()