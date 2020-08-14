#  ----- Bibliotecas ------
library(lightgbm)
library(data.table)
library(tidyverse)

# ------- Train ----
train <- fread("../data/train.csv", encoding = "UTF-8") %>% 
  select(-c(Id, property_type, operation_type, currency)) %>% 
  mutate(rooms = factor(rooms),
         bedrooms = factor(bedrooms),
         bathrooms = factor(bathrooms)) %>% 
  mutate_if(is.character, as.factor) %>% 
  relocate(price) %>% 
  as.data.frame()

# ------- Test ----
test <- fread("../data/test.csv", encoding = "UTF-8") %>% 
  select(-c(Id, property_type, operation_type, currency)) %>% 
  mutate(rooms = factor(rooms),
         bedrooms = factor(bedrooms),
         bathrooms = factor(bathrooms)) %>% 
  mutate_if(is.character, as.factor) %>% 
  as.data.frame() 

# ------- Sample  ----
sampleSub <- fread("../data/sampleSub.csv", encoding = "UTF-8")

# caret for partition data
library(caret)
set.seed(123)
indx <- createDataPartition(y = train$price, times = 1, p = 0.80,
                            list = FALSE)
dfTrain <- train[indx, ]
dfTest <- train[-indx, ]

# Categorical features
catFeatures <- names(
  train %>% select_if(is.factor)
)

# Data for lightgbm
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
  learning_rate = 0.1,
  feature_fraction = 1,
  bagging_fraction = 1
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 10000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 500)

# Best iter:307
# Best score: 70315.35

# Predictions
predicciones <- predict(modelo, data.matrix(test), 
                        num_iteration = modelo$best_iter)

predicciones[predicciones < 0] <- 0
x11();hist(predicciones)

# Submission
sampleSub %>% 
  select(Id) %>% 
  mutate(price = predicciones) %>% 
  as.data.frame() ->
  lgbmR1

# Export submission for zindi
write.csv(lgbmR1, file = "submission/lgbmR2Aptos.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  slice(1:50) %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()