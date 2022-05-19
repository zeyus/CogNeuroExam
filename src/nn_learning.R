library(tidyverse)

df <- read.csv("output/details_l_2022-05-19_19-22-17_valid.csv")

head(df)

df %>%
  ggplot(aes(x = epoch)) +
  geom_line(aes(y = loss, color = loss)) +
  scale_color_gradient(low="blue", high="red")

df %>%
  ggplot(aes(x = epoch)) +
  geom_line(aes(y = Accuracy, color = Accuracy)) +
  scale_color_gradient(low="blue", high="red")
