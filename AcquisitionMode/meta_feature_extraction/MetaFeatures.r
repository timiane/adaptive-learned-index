###################################################################
#   Complexity masures for regression problemas                   #
#   Proposed by Ana Carolina Lorena and Ivan Costa                #
#   Implemented by Aron Ifanger Maciel and Ana Carolina Lorena    #
###################################################################
library(igraph)
library(FNN)
library(moments)
library(e1071)

Normalize = function(dataset) {
  dataset = as.matrix(dataset)
  numberColumn = ncol(dataset)
  
  for (column in 1:numberColumn)
    dataset[,column] = (dataset[,column] - min(dataset[,column])) /
    (max(dataset[,column]) - min(dataset[,column]))
  
  dataset
}

FormatDataset = function(dataset, output){
  
  dataset = as.matrix(dataset)
  numberColumn = ncol(dataset)
  
  if(!is.null(output)){
    input = dataset
  } else {
    input  = as.matrix(dataset[,-numberColumn])
    output = as.matrix(dataset[,numberColumn])
    numberColumn = ncol(input)
  }
  
  list(input = Normalize(input), output = Normalize(output), 
       numberColumn = numberColumn, numberRows = nrow(input))
}

MaxPosition = function(array) order(-array)[1]

MinPosition = function(array) order(array)[1]


spearman_from_rank = function(rank){
  size=length(rank)
  results=1-6*sum(rank^2)/(size^3-size)
  results
}

ExamplesRemovedNumber = function(x,y,minCorrelation)
{
  
  numberRows = length(x)
  if(numberRows == length(y))
  {
    remainingRows = numberRows
    maxPosition = 0
    xorder=rank(x)
    yorder=rank(y)
    diff=xorder-yorder
    
    correlation = spearman_from_rank(diff)
    
    if(correlation < 0){
      yorder=rank(-y)
      diff=xorder-yorder
      correlation = spearman_from_rank(diff)
    }
    
    while(abs(correlation) < minCorrelation && !is.na(correlation))
    {
      
      maxPosition = which.max(abs(diff))
      
      diff=diff +
        ((yorder>yorder[maxPosition]) -
           (xorder>xorder[maxPosition]))
      
      yorder=yorder[-maxPosition]
      xorder=xorder[-maxPosition] 
      diff=diff[-maxPosition]
      remainingRows = remainingRows - 1
      correlation = spearman_from_rank(diff)
      
      if(is.na(correlation))
        correlation
    }
    
    (numberRows-remainingRows)/numberRows
  } else NA
}


LR = function(dataset, output = NULL){
  
  formatedDataset = FormatDataset(dataset, output)
  input = formatedDataset$input
  output = formatedDataset$output
  naRemove = !is.na(cor(output, input, method = "spearman"))
  numberColumns = formatedDataset$numberColumn
  if(numberColumns > 1)
    linearModel = lm(output~input[,naRemove])
  else 
    linearModel = lm(output~input[])
  mean(abs(linearModel$residuals))
}

OIED  = function(dataset, output = NULL){
  
  getDistances = function(dataset, numberRows, numberColumns){
    
    distances = array(0,numberRows-1)
    for(line in 2:numberRows){
      if(numberColumns > 1)
        distances[line-1] = dist(dataset[(line-1):line,])
      else
        distances[line-1] = dist(dataset[(line-1):line])
    } 
    distances
  }
  
  formatedDataset = FormatDataset(dataset, output)
  input = formatedDataset$input
  output = formatedDataset$output
  numberRows = formatedDataset$numberRows
  numberColumns = formatedDataset$numberColumn
  
  order = order(output)
  if(numberColumns > 1)
    distances = getDistances(input[order,], numberRows,numberColumns)
  else
    distances = getDistances(input[order], numberRows,numberColumns)
  
  mean(distances)
}

SNNR = function(dataset, output = NULL){
  
  formatedDataset = FormatDataset(dataset, output)
  input = formatedDataset$input
  output = formatedDataset$output
  numberRows = formatedDataset$numberRows
  naRemove = !is.na(cor(output, input, method = "spearman"))
  numberColumn = sum(naRemove)
  
  order = order(output)
  output = output[order,] 
  numberColumns = formatedDataset$numberColumn
  randomUniform = runif(numberRows - 1)
  
  if(numberColumns > 1){
    input = input[order,]  
    newInput = randomUniform*input[2:numberRows-1,] + 
      (1-randomUniform)*input[2:numberRows,]
    
  }
  else{
    input = input[order]
    newInput = randomUniform*input[2:numberRows-1] + 
      (1-randomUniform)*input[2:numberRows]
    
  }
  newOutput = randomUniform*output[2:numberRows-1] + 
    (1-randomUniform)*output[2:numberRows]
  newPredict = knn.reg(as.data.frame(input), as.data.frame(newInput), output, k = 1)$pred
  mean((newPredict - newOutput)^2)
}

Inst = function(dataset, output = NULL){
  formatedDataset = FormatDataset(dataset, output)
  (formatedDataset$numberRows / formatedDataset$numberColumn)
}


kurt = function(dataset, output=NULL){
  formatedDataset = FormatDataset(dataset, output)
  kurtosis(formatedDataset$input)
}

skew = function(dataset, output=NULL) {
  formatedDataset = FormatDataset(dataset, output)
  skewness(formatedDataset$input)
}

variance = function(dataset, output=NULL) {
  formatedDataset = FormatDataset(dataset, output)
  stats::var(formatedDataset$input)
}

deltaDifference = function(dataset, output=NULL){
  formatedDataset = FormatDataset(dataset, output)
  meanDelta = 1 / length(dataset)
  differece <- diff(formatedDataset$input)
  result = 0
  for(datapoint in differece) {
    if(abs(datapoint) > meanDelta){
      result <- result + 1
    }
  }
  result
}

correlation = function(dataset, output=NULL) {
  formatedDataset = FormatDataset(dataset, output)
  cor(formatedDataset$input, formatedDataset$output)
}

titles = function(header, output = NULL){
  result = ""
  if(grepl("logn", header)){
    result = "logn"}
  if(grepl("beta", header)){
    result = "beta"}
  if(grepl("sinl", header)){
    result = "sinl"}
  if(grepl("parteo", header)){
    result = "parteo"}
  if(grepl("wald", header)){
    result = "wald"}
  if(grepl("logis", header)){
    result = "logis"}
  if(grepl("uni", header)){
    result = "uni"}
  result
}

autoMeta = function(path){
  filenames <- list.files(path, pattern = "*.csv", full.names = TRUE)
  headers <- list.files(path, pattern = "*.csv", full.names = FALSE)
  ldf <- sapply(filenames, read.csv)
  metaFeatures = list()
  i <- 1
  for (dataset in ldf) {
    result <- data.frame(x = 1:length(dataset))
    #dataset = dataset[order(dataset$V1), ]

    temp <- data.frame(LR(dataset, result),
                       #OIED(dataset, result),
                       #L3(dataset, result),
                       SNNR(dataset, result),
                       Inst(dataset, result),
                       variance(dataset, result),
                       deltaDifference(dataset, result), 
                       kurt(dataset, result),
                       correlation(dataset, result),
                       headers[i]
                      
    )
                       
    
    metaFeatures[[i]] <- temp
    i <- i + 1
  }
  output = do.call(rbind, metaFeatures)
  write.csv(output, file = "metaFeatures.csv", sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
}
setwd(path)
print("Extracting meta-features")
autoMeta(path)
print("Meta-features extracted")
