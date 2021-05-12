library(causalToolbox,lib = "/home/fg746/R/4.0.4")
library(Rforestry,lib = "/home/fg746/R/4.0.4")
library(dpylr,lib = "/home/fg746/R/4.0.4")
setwd("/scratch/fg746/meta-learners")
gotv<-read.csv("gotv.csv")
x_vars <- select(gotv, sex, g2000, g2002, p2000, p2002, p2004, age)
X_0 <- x_vars[gotv$treatment == 0,]
X_1 <- x_vars[gotv$treatment == 1,]
yobs_0 <- gotv$voted[gotv$treatment == 0]
yobs_1 <- gotv$voted[gotv$treatment == 1]
filename <- paste0("results.csv")

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
results<- data.frame(TRF = (predict(m_1, x_vars) - predict(m_0, x_vars)))


m <- forestry(x = cbind(x_vars, gotv$treatment),
              y = gotv$voted,
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
results<- cbind(results,data.frame(SRF =  (predict(m,cbind(x_vars, rep.int(1, nrow(gotv)))) - predict(m,cbind(x_vars, rep.int(0, nrow(gotv)))))))


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
                   y= gotv$treatment,
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

prop_scores <- predict(m_prop,x_vars)
results<- cbind(results,data.frame(XRF = (prop_scores * predict(m_tau_0, x_vars) + (1-prop_scores) * predict(m_tau_1,x_vars))))

write.csv(results,
          file = filename,
          row.names = FALSE)
