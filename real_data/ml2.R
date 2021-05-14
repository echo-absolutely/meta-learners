library(causalToolbox,lib = "/home/fg746/R/4.0.4")
library(Rforestry,lib = "/home/fg746/R/4.0.4")
library(dplyr,lib = "/home/fg746/R/4.0.4")

setwd("/scratch/fg746/meta-learners")
gotv<-read.csv("gotv.csv")
training_sizes<- c(100, 500, 1000, 2000, 5000, 15000, 40000, 80000)
count<- 0
for (i in training_sizes){
  count = count + 1
  idx<- sample(nrow(gotv),i)
  train_data <- gotv[idx,]
  test_data<- gotv[-idx,]
  w_train<-train_data$treatment
  results<- data.frame(indices = rownames(test_data))
  x_vars <- select(train_data, sex, g2000, g2002, p2000, p2002, p2004, age)
  X_0 <- x_vars[train_data$treatment == 0,]
  X_1 <- x_vars[train_data$treatment == 1,]
  yobs_0 <- train_data$voted[train_data$treatment == 0]
  yobs_1 <- train_data$voted[train_data$treatment == 1]
  x_vars_test <- select(test_data, sex, g2000, g2002, p2000, p2002, p2004, age)
  filename <- paste0("results",toString(count),".csv")
  
  m_0 <- forestry(x = X_0,
                  y= yobs_0,
                  ntree = 1000,
                  replace = TRUE,
                  sample.fraction = 0.9,
                  mtry = ncol(X_0),
                  nodesizeSpl = 1,
                  nodesizeAvg = 3,
                  nodesizeStrictSpl = 1,
                  nodesizeStrictAvg = 1,
                  nthread = 0,
                  splitrule = 'variance',
                  splitratio = 1,
                  middleSplit = FALSE,
                  OOBhonest = TRUE)
  m_1 <- forestry(x = X_1,
                  y= yobs_1,
                  ntree = 1000,
                  replace = TRUE,
                  sample.fraction = 0.9,
                  mtry = ncol(X_0),
                  nodesizeSpl = 1,
                  nodesizeAvg = 3,
                  nodesizeStrictSpl = 1,
                  nodesizeStrictAvg = 1,
                  nthread = 0,
                  splitrule = 'variance',
                  splitratio = 1,
                  middleSplit = FALSE,
                  OOBhonest = TRUE)
  results<- cbind(results,TRF = (predict(m_1, x_vars_test) - predict(m_0, x_vars_test)))
  print("T done")
  
  m <- forestry(x = cbind(x_vars, w_train),
                y = train_data$voted,
                ntree = 1000,
                replace = TRUE,
                sample.fraction = 0.9,
                mtry = ncol(x_vars),
                nodesizeSpl = 1,
                nodesizeAvg = 3,
                nodesizeStrictSpl = 3,
                nodesizeStrictAvg = 1,
                nthread = 0,
                splitrule = "variance",
                splitratio = 1,
                middleSplit = FALSE,
                OOBhonest = TRUE)
  results<- cbind(results,SRF = (predict(m,cbind(x_vars_test, w_train=1))) - predict(m,cbind(x_vars_test, w_train=0)))
  print("S done")
  m_0_xrf <- forestry(x = X_0,
                      y= yobs_0,
                      ntree = 1000,
                      replace = TRUE,
                      sample.fraction = 0.8,
                      mtry = round(ncol(X_0)*13/20),
                      nodesizeSpl = 2,
                      nodesizeAvg = 1,
                      nodesizeStrictSpl = 2,
                      nodesizeStrictAvg = 1,
                      nthread = 0,
                      splitrule = 'variance',
                      splitratio = 1,
                      middleSplit = TRUE,
                      OOBhonest = TRUE)
  m_1_xrf <- forestry(x = X_1,
                      y= yobs_1,
                      ntree = 1000,
                      replace = TRUE,
                      sample.fraction = 0.8,
                      mtry = round(ncol(X_0)*13/20),
                      nodesizeSpl = 2,
                      nodesizeAvg = 1,
                      nodesizeStrictSpl = 2,
                      nodesizeStrictAvg = 1,
                      nthread = 0,
                      splitrule = 'variance',
                      splitratio = 1,
                      middleSplit = TRUE,
                      OOBhonest = TRUE)
  r_0 <- predict(m_1_xrf, X_0) - yobs_0
  r_1 <- yobs_1 -  predict(m_0_xrf, X_1)
  
  m_tau_0 <- forestry(x = X_0,
                      y= r_0,
                      ntree = 1000,
                      replace = TRUE,
                      sample.fraction = 0.7,
                      mtry = round(ncol(X_0)*17/20),
                      nodesizeSpl = 5,
                      nodesizeAvg = 6,
                      nodesizeStrictSpl = 3,
                      nodesizeStrictAvg = 1,
                      nthread = 0,
                      splitrule = 'variance',
                      splitratio = 1,
                      middleSplit = TRUE,
                      OOBhonest = TRUE)
  m_tau_1 <- forestry(x = X_1,
                      y= r_1,
                      ntree = 1000,
                      replace = TRUE,
                      sample.fraction = 0.7,
                      mtry = round(ncol(X_0)*17/20),
                      nodesizeSpl = 5,
                      nodesizeAvg = 6,
                      nodesizeStrictSpl = 3,
                      nodesizeStrictAvg = 1,
                      nthread = 0,
                      splitrule = 'variance',
                      splitratio = 1,
                      middleSplit = TRUE,
                      OOBhonest = TRUE)
  
  m_prop <- forestry(x = x_vars,
                     y= train_data$treatment,
                     ntree = 500,
                     replace = TRUE,
                     sample.fraction = 0.5,
                     mtry = ncol(x_vars),
                     nodesizeSpl = 11,
                     nodesizeAvg = 33,
                     nodesizeStrictSpl = 2,
                     nodesizeStrictAvg = 1,
                     nthread = 0,
                     splitrule = 'variance',
                     splitratio = 1,
                     middleSplit = FALSE,
                     OOBhonest = TRUE)
  
  prop_scores <- predict(m_prop,x_vars_test)
  results<- cbind(results,XRF = (prop_scores * predict(m_tau_0, x_vars_test) + (1-prop_scores) * predict(m_tau_1,x_vars_test)))
  print("X done")
  write.csv(results,
            file = filename,
            row.names = FALSE)
}
