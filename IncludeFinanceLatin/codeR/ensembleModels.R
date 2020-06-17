# All predictions
lgbm1 <- read.csv("submission/lgbm1.csv")
lgbm2 <- read.csv("submission/lgbm3.csv")
lgbm3 <- read.csv("submission/lgbm4.csv")
catb1 <- read.csv("submission/catB1.csv")
catb2 <- read.csv("submission/catB2.csv")
catb3 <- read.csv("submission/catB3.csv")
catb4 <- read.csv("submission/catB4.csv")
catb5 <- read.csv("submission/catB5.csv")
catb6 <- read.csv("submission/catB6.csv")

# Join data
oneData <- data.frame(
  uniqueid = lgbm1$uniqueid,
  lgbm1 = lgbm1$bank_account,
  lgbm2 = lgbm2$bank_account,
  lgbm3 = lgbm3$bank_account,
  catb1 = catb1$bank_account,
  catb2 = catb2$bank_account,
  catb3 = catb3$bank_account,
  catb4 = catb4$bank_account,
  catb5 = catb5$bank_account,
  catb6 = catb6$bank_account
)

# One prediction
dataEnsemble <- data.frame(
  uniqueid = oneData$uniqueid,
  bank_account = ifelse(apply(oneData[, -1], 1, sum) >= 5, "1", "0")
)

write.csv(dataEnsemble, file = "submission/ensemble.csv", row.names = FALSE)
