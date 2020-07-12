library(data.table)
train <- fread("data/train.csv", encoding = "UTF-8")
test <- fread("data/test.csv", encoding = "UTF-8")
sampleSub <- fread("data/sampleSub.csv", encoding = "UTF-8")
