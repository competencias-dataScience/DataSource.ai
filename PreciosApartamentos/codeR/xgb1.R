#  ----- Bibliotecas ------
library(data.table)
library(tidyverse)
library(h2o)

# ------- Test ----
load("newTest1.Rdata")

# ------- Train ----
load("newTrain1.Rdata")

# ------- Sample  ----
sampleSub <- fread("../data/sampleSub.csv", encoding = "UTF-8")

# Iniciando h2o
h2o.init(nthreads = -1, port = 54321, max_mem_size = "3g")

# df_train y df_test
df_train <- newTrain1 %>% 
  relocate(price, everything())
df_test <- newTest1

datos_h2o <- as.h2o(x = df_train, destination_frame = "datos_h2o")
particiones <- h2o.splitFrame(data = datos_h2o, ratios = c(0.8, 0.10),
                              seed = 123)
datos_train_h2o <- h2o.assign(data = particiones[[1]], key = "datos_train_h2o")
datos_val_h2o   <- h2o.assign(data = particiones[[2]], key = "datos_val_h2o")
datos_test_h2o  <- h2o.assign(data = particiones[[3]], key = "datos_test_h2o")

# Variable respuesta y predictores
var_respuesta <- "price"
predictores   <- setdiff(h2o.colnames(datos_train_h2o), var_respuesta)

xgb <- h2o.xgboost(
  # General
  x = predictores,
  y = var_respuesta,
  training_frame = datos_train_h2o,
  validation_frame = datos_val_h2o,
  model_id = "xgb1"  ,
  stopping_metric = "RMSE",
  distribution = "gaussian",
  
  # Learning
  learn_rate = 0.1,
  ntrees = 5000,
  verbose = 1,
  stopping_rounds = 50,
  max_depth = 0,
  tree_method = "hist",
  grow_policy = "lossguide",
  booster = "gbtree",
  
  # Cross validation
  nfolds = 5,
  seed = 123,
  max_runtime_secs = 3600
)

# Resultados del modelo
xgb

# Train
predichos_train <- h2o.predict(xgb, datos_train_h2o) %>%
  as.data.frame() %>% pull(predict)

# Test (Train)
predichos_test <- h2o.predict(xgb, datos_test_h2o) %>%
  as.data.frame() %>% pull(predict)

# Test (Submission)
predichos_subm <- h2o.predict(xgb, as.h2o(df_ancha_test)) %>%
  as.data.frame() %>% pull(predict)

# Submission
sampleSub %>% 
  select(Id) %>% 
  mutate(price = predicciones) %>%
  rename(id = Id) %>% 
  as.data.frame() ->
  lgbmR14

# Export submission for zindi
write.csv(lgbmR14, file = "submission/lgbmR14Aptos.csv", row.names = FALSE)

df_ancha_test %>% 
  select(ID) %>% 
  mutate(target = predichos_subm) ->
  subm11

# Exportando predicciones
write.csv(subm11, file = "subm11.csv", row.names = FALSE)