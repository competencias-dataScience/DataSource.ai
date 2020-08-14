sm6 <- read.csv("Submission/lgbmR6Aptos.csv")
sm7 <- read.csv("Submission/lgbmR7Aptos.csv")
sm8 <- read.csv("Submission/lgbmR8Aptos.csv")

library(tidyverse)
s10 <- data.frame(Id = sm6$Id, sm6 = sm6$price, sm7 = sm7$price, sm8 = sm8$price) %>% 
  mutate(price = (sm6 + sm7 + sm8)/3) %>% 
  select(-c(sm6:sm8))

# Export submission for zindi
write.csv(s10, file = "submission/ensamble1.csv", row.names = FALSE)
