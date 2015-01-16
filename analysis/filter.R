################
## load packages
################

library(plyr)

############
## load data
############

root <- '~/experiments/ImplicativeExperiments/'

data <- read.csv(paste(root, "data/results.preprocessed", sep=''))
data.test <- subset(data, itemtype=='test')

###################
## exclude subjects
###################


## calculate first, second (median), and third quartiles of log(RT)
subj.rt.stats <- ddply(data.test, .(subj), summarise, medianlogrt=median(log(rt)), firstlogrt=quantile(log(rt), .25), thirdlogrt=quantile(log(rt), .75), iqrlogrt=IQR(log(rt)))

## calculate upper and lower 1.5*IQR for Tukey method exclusion
subj.rt.stats$tukeylow <- subj.rt.stats$firstlogrt - 1.5*subj.rt.stats$iqrlogrt
subj.rt.stats$tukeyhigh <- subj.rt.stats$thirdlogrt + 1.5*subj.rt.stats$iqrlogrt

## calculate first and second quartiles and IQR of median log(RT)
## (how fast is the subject overall?)
median.medianlogrt <- median(subj.rt.stats$medianlogrt)
lower.medianlogrt <- quantile(subj.rt.stats$medianlogrt, .25)
iqr.medianlogrt <- IQR(subj.rt.stats$medianlogrt)

## remove subjects below (faster than) lower Tukey bound
subj.rt.stats <- subset(subj.rt.stats, medianlogrt > (lower.medianlogrt - 1.5*iqr.medianlogrt))

## calculate first, second, and third quartiles and IQR of log(RT) IQR
## (how much does the subject's speed vary?)
median.iqrlogrt <- median(subj.rt.stats$iqrlogrt)
lower.iqrlogrt <- quantile(subj.rt.stats$iqrlogrt, .25)
upper.iqrlogrt <- quantile(subj.rt.stats$iqrlogrt, .75)
iqr.iqrlogrt <- IQR(subj.rt.stats$iqrlogrt)

## remove subjects below (less variable than expected) lower Tukey bound and above (more variable than expected) upper Tukey bound
subj.rt.stats <- subset(subj.rt.stats, iqrlogrt > (lower.iqrlogrt - 1.5*iqr.iqrlogrt))
subj.rt.stats <- subset(subj.rt.stats, iqrlogrt < (upper.iqrlogrt + 1.5*iqr.iqrlogrt))

## merge subjects that passed the filter with data 
## (removes filtered subjects)
data.test <- merge(data.test, subj.rt.stats)

#######################
## exclude observations
#######################

## remove observations that are significantly faster than the median for the subject 
data.test <- subset(data.test, log(rt) > tukeylow & log(rt) < tukeyhigh)

################################
##  map response to hardness
################################

## (positive & must be true) | (negative & must be false) -> positive
data.test$hardness <- ifelse((data.test$followup == 'positive' & data.test$response == 'Must be true') | (data.test$followup == 'negative' & data.test$response == 'Must be false'), 'positive', 'soft')

## (positive & must be false) | (negative & must be true) -> negative
data.test$hardness <- ifelse((data.test$followup == 'positive' & data.test$response == 'Must be false') | (data.test$followup == 'negative' & data.test$response == 'Must be true'), 'negative', data.test$hardness)

## reorder hardness levels
data.test$hardness <- ordered(data.test$hardness, levels=c('negative', 'soft', 'positive'))

#############
## write data
#############

## uncomment to write data
write.csv(data.test[c('subj', 'item', 'verb', 'negation', 'hardness', 'rt')], '~/experiments/ImplicativeExperiments/results.filtered', quote=F, row.names=F)