library(corrplot)
data <- read.csv(file="metaFeaturesCC2.csv")
colnames(data) <- c('CC', 'LR', 'SNNR', 'Inst', 'var', 'delta', 'kurt', 'cor')
data = data[-9]
correlations = cor(data)
corrplot(correlations, method = "circle",type = "upper" )