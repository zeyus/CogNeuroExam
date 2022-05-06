library(tidyverse)

df <- read.csv("output/details_a_2022-05-05_10-22-12_valid.csv")

head(df)

df %>% filter(epoch > 5500) %>% filter(epoch < 5750) %>%
  ggplot(aes(x = epoch)) +
  geom_line(aes(y = loss), color = "red") +
  geom_line(aes(y = Accuracy), color = "blue") +
  geom_point(aes(y = loss)) +
  geom_point(aes(y = Accuracy)) +
  labs(title = "Loss (red) and Accuracy (blue) by Epoch")
