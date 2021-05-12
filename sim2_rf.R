library(causalToolbox,lib = "/home/fg746/R/4.0.4")
library(Rforestry,lib = "/home/fg746/R/4.0.4")

n_range <- c(5000, 10000, 20000, 100000, 200000, 300000)
N <- 300000
n_range <- n_range[which(n_range <= N)]
results <- data.frame(N = n_range)
results$XRF <- NA
results$SRF <- NA
results$TRF <- NA
filename <- paste0("sim2_rf.csv")

for (n in n_range) {
  exp <- simulate_causal_experiment(ntrain = n,
                                    ntest = 1000,
                                    dim = 20,
                                    pscore = "rct5",
                                    mu0 = "fullLinearStrong",
                                    tau = "fullLinearStrong")
  
  feature_train <- exp$feat_tr
  w_train <- exp$W_tr
  yobs_train <- exp$Yobs_tr
  X_0 <- feature_train[w_train == 0,]
  X_1 <- feature_train[w_train == 1,]
  yobs_0 <- yobs_train[w_train == 0]
  yobs_1 <- yobs_train[w_train == 1]
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
  
  feature_test <- exp$feat_te
  cate_true <- exp$tau_te
  cate_esti_trf = predict(m_1, feature_test) - predict(m_0, feature_test)
  results_trf <- mean((cate_esti_trf - cate_true)^2)
  results$TRF[which(results$N == n)] <- results_trf
  
  feature_train <- exp$feat_tr
  w_train <- exp$W_tr
  yobs_train <- exp$Yobs_tr
  X_0 <- feature_train[w_train == 0,]
  X_1 <- feature_train[w_train == 1,]
  yobs_0 <- yobs_train[w_train == 0]
  yobs_1 <- yobs_train[w_train == 1]
  
  m <- forestry(x = cbind(feature_train, w_train),
                y = yobs_train,
                ntree = 1000,
                replace = TRUE,
                sample.fraction = 0.9,
                mtry = ncol(feature_train),
                nodesizeSpl = 1,
                nodesizeAvg = 3,
                nodesizeStrictSpl = 3,
                nodesizeStrictAvg = 1,
                nthread = 0,
                splitrule = "variance",
                splitratio = 1,
                middleSplit = FALSE,
                OOBhonest = TRUE)
  
  cate_est_srf <- predict(m,cbind(feature_test, w_train = 1)) - predict(m,cbind(feature_test, w_train = 0)) 
  results_srf <- mean((cate_est_srf - cate_true)^2)
  results$SRF[which(results$N == n)] <- results_srf
  
  feature_train <- exp$feat_tr
  w_train <- exp$W_tr
  yobs_train <- exp$Yobs_tr
  X_0 <- feature_train[w_train == 0,]
  X_1 <- feature_train[w_train == 1,]
  yobs_0 <- yobs_train[w_train == 0]
  yobs_1 <- yobs_train[w_train == 1]
  
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
  
  m_prop <- forestry(x = feature_train, 
                     y= w_train, 
                     ntree = 500, 
                     replace = TRUE,  
                     sample.fraction = 0.5, 
                     mtry = ncol(feature_train), 
                     nodesizeSpl = 11, 
                     nodesizeAvg = 33, 
                     nodesizeStrictSpl = 2, 
                     nodesizeStrictAvg = 1, 
                     nthread = 0, 
                     splitrule = 'variance',  
                     splitratio = 1, 
                     middleSplit = FALSE, 
                     OOBhonest = TRUE)
  
  prop_scores <- predict(m_prop,feature_test)
  cate_est_xrf <- prop_scores * predict(m_tau_0, feature_test) + (1-prop_scores) * predict(m_tau_1, feature_test)
  results_xrf <- mean((cate_est_xrf - cate_true)^2)
  results$XRF[which(results$N == n)] <- results_xrf
  
  write.csv(results,
            file = filename,
            row.names = FALSE)
}
