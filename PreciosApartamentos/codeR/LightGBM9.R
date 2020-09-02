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

test_dummy <- dummy_cols(test)

# ------- Train ----
load("newTrain1.Rdata")
train <- newTrain1 %>% 
  mutate(rooms = factor(rooms),
         bedrooms = factor(bedrooms),
         bathrooms = factor(bathrooms)) %>% 
  mutate_if(is.character, as.factor) %>% 
  relocate(price) %>% 
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
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -246]),
                              label = dfTrain[, 246])
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -246]),
                             label = dfTest[, 246])

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "poisson",
  metric = "rmse",
  learning_rate = 0.01,
  feature_fraction = 1,
  bagging_fraction = 1
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 10000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 500)

# Best iter:6873
# Best score: 66961.29

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
  lgbmR9

# Export submission for zindi
write.csv(lgbmR9, file = "submission/lgbmR9Aptos.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  slice(1:50) %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()