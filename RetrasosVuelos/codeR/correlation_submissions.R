# Submission data
s1 = read.csv("../submission/lgbmR1Vuelos.csv")
s2 = read.csv("../submission/lgbmR2Vuelos.csv")
s3 = read.csv("../submission/lgbmR3Vuelos.csv")
s4 = read.csv("../submission/lgbmR4Vuelos.csv")
s5 = read.csv("../submission/lgbmR5Vuelos.csv")
s6 = read.csv("../submission/lgbmR6Vuelos.csv")
s7 = read.csv("../submission/lgbmR7Vuelos.csv")
s8 = read.csv("../submission/lgbmR8Vuelos.csv")

# Data
dataCor <- data.frame(
  s1 = s1$target, s2 = s2$target, s3 = s3$target,
  s4 = s4$target, s5 = s5$target, s6 = s6$target,
  s7 = s7$target, s8 = s8$target
)

# Correlation graphic --> Pearson
library(corrplot)
x11()
corrplot(
  corr = cor(dataCor, method = "pearson"),
  type = "upper",
  diag = FALSE,
  method = "pie",
  col = RColorBrewer::brewer.pal(n = 8, name = "Spectral")
)


# Correlation graphic --> Spearman
x11()
corrplot(
  corr = cor(dataCor, method = "spearman"),
  type = "upper",
  diag = FALSE,
  method = "pie",
  col = RColorBrewer::brewer.pal(n = 8, name = "Spectral")
)

