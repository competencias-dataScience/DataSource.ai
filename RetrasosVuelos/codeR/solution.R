library(data.table)
dataTrain <- fread("../data/train.csv")
dataTest <- fread("../data/test.csv")
dataSample <- fread("../data/sample.csv")

library(tidyverse)
dataTrain
