xgb <- read.csv("sub_09_xgboost.csv")
catb <- read.csv("sub_17_catboost_boosting.csv")
gbm1 <- read.csv("sub_18_gbm_tuned.csv")
gbm2 <- read.csv("sub_19_gbm_bagging.csv")
gbm3 <- read.csv("sub_20_gbm_boosting.csv")

table(gbm3$rating)
table(gbm1$rating)
table(gbm2$rating)

cat1 <- read.csv("sub_15_catboost_tuned.csv")
table(cat1$rating)

cat2 <- read.csv("sub_16_catboost_bagging.csv")
table(cat2$rating)

cat3 <- read.csv("sub_17_catboost_boosting.csv")
table(cat3$rating)

lgbm1 <- read.csv("sub_21_lgbm_tuned.csv")
lgbm2 <- read.csv("sub_22_lgbm_bagging.csv")
lgbm3 <- read.csv("sub_23_lgbm_boosting.csv")

table(lgbm1$rating)
table(lgbm2$rating)
table(lgbm3$rating)

table(lgbm1$rating, xgb$rating)

table(catb$rating, xgb$rating)
table(catb$rating, lgbm1$rating)


ensamble <- read.csv("sub_11_ensamble_svmRRFXGB.csv")
table(ensamble$rating, catb$rating)


new_xgb <- read.csv("sub_24_xgb_tuned.csv")
new_xgb2 <- read.csv("sub_25_xgb_bagging.csv")
table(new_xgb$rating)
table(new_xgb2$rating)


catboost_500iter <- read.csv("sub_31_catboost_tuned.csv")
table(catboost_500iter$rating)

catboost_500iter_boost <- read.csv("sub_32_catboost_boosting.csv")
table(catboost_500iter_boost$rating)

prueba <- read.csv("sub_17_catboost_boosting.csv")
table(prueba$rating)

table(prueba$rating, catboost_500iter_boost$rating)


# Pruebas con xgboost (R - F1), smvR, catboost (boosting 1 - AUC) y
# Catboost (Boosting 2 - F1)
xgb <- read.csv("sub_09_xgboost.csv")
svm <- read.csv("sub_03_svmR.csv")
cat1 <- read.csv("sub_17_catboost_boosting.csv")
cat2 <- read.csv("sub_32_catboost_boosting.csv")


cat_tuned <- read.csv("sub_37_catboost_tuned.csv")
table(cat_tuned$rating)

cat_boost <- read.csv("sub_38_catboost_boosting.csv")
table(cat_boost$rating)

table(cat_boost$rating, cat_tuned$rating)
