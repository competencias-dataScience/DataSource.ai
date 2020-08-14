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
  as.data.frame()

train_dummy <- dummy_cols(train)
train_dummy2 <- train_dummy %>% 
  select_at(vars(matches(names(test_dummy))))

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
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -237]),
                              label = dfTrain[, 237])
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -237]),
                             label = dfTest[, 237])

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.1,
  feature_fraction = 0.5,
  bagging_fraction = 1,
  min_data_in_leaf = 100,
  num_leaves = 255,
  max_depth = -1
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 10000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 500)

# Best iter:811
# Best score: 74066.53

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
  lgbmR3

# Export submission for zindi
write.csv(lgbmR3, file = "submission/lgbmR3Aptos.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  slice(1:50) %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()