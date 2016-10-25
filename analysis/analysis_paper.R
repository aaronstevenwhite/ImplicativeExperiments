library(dplyr)
library(ordinal)
library(ggplot2)
library(lme4)
library(tikzDevice)

set.seed(2653)

theme_set(theme_bw()+theme(panel.grid=element_blank()))

root <- '~/experiments/ImplicativeExperiments/'

ridit <- function(x){
  tab <- table(x)
  tot <- sum(tab)
  prop <- tab/tot
  
  cum <- list()
  sigma <- 0
  for (n in rownames(prop)){
    cum[n] <- prop[n]*.5 + sigma
    sigma <- sigma + prop[n]
  }
  
  out <- c()
  
  for (val in x){
    out <- c(out, cum[[val]])
  }
  
  return(out)
}

filter.subjects <- function(d) {
  subj.rt.stats <- summarise(group_by(d, subj), 
                             medianlogrt=median(log(rt)), 
                             q1logrt=quantile(log(rt), .25), 
                             q3logrt=quantile(log(rt), .75), 
                             iqrlogrt=IQR(log(rt)),
                             tukeylow=q1logrt - 1.5*iqrlogrt,
                             tukeyhigh=q3logrt + 1.5*iqrlogrt)
  
  subj.rt.stats <- mutate(subj.rt.stats,
                          median.medianlogrt=median(medianlogrt),
                          q1.medianlogrt=quantile(medianlogrt, .25),
                          iqr.medianlogrt=IQR(subj.rt.stats$medianlogrt),
                          median.iqrlogrt=median(iqrlogrt),
                          q1.iqrlogrt=quantile(iqrlogrt, .25),
                          q3.iqrlogrt=quantile(iqrlogrt, .75),
                          iqr.iqrlogrt=IQR(iqrlogrt))

  n.median.filtered <- nrow(filter(subj.rt.stats, 
                                   medianlogrt < (q1.medianlogrt - 1.5*iqr.medianlogrt)))
    
  subj.rt.stats.medianfiltered <- filter(subj.rt.stats, 
                                         medianlogrt > (q1.medianlogrt - 1.5*iqr.medianlogrt))
  
  n.iqr.filtered <- nrow(filter(subj.rt.stats, 
                                iqrlogrt < (q1.iqrlogrt - 1.5*iqr.iqrlogrt) |
                                iqrlogrt > (q3.iqrlogrt + 1.5*iqr.iqrlogrt)))
  
  subj.rt.stats.iqrfiltered <- filter(subj.rt.stats.medianfiltered, 
                                      iqrlogrt > (q1.iqrlogrt - 1.5*iqr.iqrlogrt),
                                      iqrlogrt < (q3.iqrlogrt + 1.5*iqr.iqrlogrt))
  
  cat('median filtered', n.median.filtered, '\n')
  cat('IQR filtered', n.iqr.filtered, '\n')
  
  d <- merge(d, subj.rt.stats.iqrfiltered)
  
  d <- filter(d, log(rt) > tukeylow)
  
  return(d)
}

data.entailment <- read.csv(paste(root, "data/entailment/results.preprocessed", sep=''))
data.entailment.filtered <- filter.subjects(filter(data.entailment, itemtype=='test'))

data.implicature <- read.csv(paste(root, "data/implicature/results.preprocessed", sep=''))
data.implicature.filtered <- filter.subjects(filter(data.implicature, itemtype=='test'))

data.likelihood <- read.csv(paste(root, "data/likelihood/results.preprocessed", sep=''))
data.likelihood.filtered <- filter.subjects(filter(data.likelihood, itemtype=='test'))


data.likelihood.filtered$response.factor <- as.factor(data.likelihood.filtered$response)
data.likelihood.filtered <- group_by(data.likelihood.filtered, subj) %>% mutate(response.ridit=ridit(response.factor))

## reorder verbs 
data.entailment.filtered$verb.ordered <- ordered(data.entailment.filtered$verb, levels=c('manage', 'opt', 'remember', 'think', 'know', 'want', 'hope', 'hasten', 'hesitate', 'mean', 'refuse', 'forget', 'neglect', 'fail'))
data.implicature.filtered$verb.ordered <- ordered(data.implicature.filtered$verb, levels=c('manage', 'opt', 'remember', 'think', 'know', 'want', 'hope', 'hasten', 'hesitate', 'mean', 'refuse', 'forget', 'neglect', 'fail'))
data.likelihood.filtered$verb.ordered <- ordered(data.likelihood.filtered$verb, levels=c('manage', 'opt', 'remember', 'think', 'know', 'want', 'hope', 'hasten', 'hesitate', 'mean', 'refuse', 'forget', 'neglect', 'fail'))


##
levels(data.entailment.filtered$frame) <- c('NP Ved not to VP', 'NP Ved to not VP', 'NP didn\'t V to VP', 'NP Ved to VP')
data.entailment.filtered$frame.ordered <- ordered(data.entailment.filtered$frame, levels=c('NP Ved to VP', 'NP didn\'t V to VP', 'NP Ved not to VP', 'NP Ved to not VP'))

levels(data.implicature.filtered$frame) <- c('NP Ved not to VP', 'NP Ved to not VP', 'NP didn\'t V to VP', 'NP Ved to VP')
data.implicature.filtered$frame.ordered <- ordered(data.implicature.filtered$frame, levels=c('NP Ved to VP', 'NP didn\'t V to VP', 'NP Ved not to VP', 'NP Ved to not VP'))

levels(data.likelihood.filtered$frame) <- c('NP Ved not to VP', 'NP Ved to not VP', 'NP didn\'t V to VP', 'NP Ved to VP')
data.likelihood.filtered$frame.ordered <- ordered(data.likelihood.filtered$frame, levels=c('NP Ved to VP', 'NP didn\'t V to VP', 'NP Ved not to VP', 'NP Ved to not VP'))

##
positive.positive <- data.entailment.filtered$followup == 'positive' & data.entailment.filtered$response == 'Must be true'
negative.negative <- data.entailment.filtered$followup == 'negative' & data.entailment.filtered$response == 'Must be false'

data.entailment.filtered$entailment.polarity <- ifelse(positive.positive | negative.negative, 
                                        'positive', 
                                        'none')

positive.negative <- data.entailment.filtered$followup == 'positive' & data.entailment.filtered$response == 'Must be false'
negative.positive <- data.entailment.filtered$followup == 'negative' & data.entailment.filtered$response == 'Must be true'

data.entailment.filtered$entailment.polarity <- ifelse(positive.negative | negative.positive, 
                                        'negative', 
                                        data.entailment.filtered$entailment.polarity)

## reorder entailment.polarity levels
data.entailment.filtered$entailment.polarity.ordered <- ordered(data.entailment.filtered$entailment.polarity, levels=c('negative', 'none', 'positive'))
data.implicature.filtered$response <- ordered(data.implicature.filtered$response, levels=c('No', 'Maybe or maybe not', 'Yes'))

## plot
#tikz('~/experiments/ImplicativeExperiments/analysis/plots/entailment.tikz', width=5.5, height=5)
ggplot(data.entailment.filtered, aes(x=verb.ordered, fill=entailment.polarity.ordered)) + geom_bar(position="fill") + facet_grid(frame.ordered~.) + scale_x_discrete(name='') + scale_y_continuous() + scale_fill_manual(name=element_blank(), values = c("#8c510a", "#f5f5f5", "#01665e"), labels=c('-', '', '+')) + theme(axis.text.x=element_text(size=12,angle = 45, hjust=1), axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank(), legend.text=element_text(size=15))
#dev.off()

#tikz('~/experiments/ImplicativeExperiments/analysis/plots/implicature.tikz', width=5.5, height=5)
ggplot(data.implicature.filtered, aes(x=verb.ordered, fill=response)) + geom_bar(position="fill") + facet_grid(frame.ordered~.) + scale_x_discrete(name='') + scale_fill_manual(name=element_blank(), values = c("#8c510a", "#f5f5f5", "#01665e"), labels=c('-', '', '+')) + theme(axis.text.x=element_text(size=12,angle = 45, hjust=1), axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank(), legend.text=element_text(size=15))
#dev.off()

#tikz('~/experiments/ImplicativeExperiments/analysis/plots/likelihood.tikz', width=5.5, height=5)
ggplot(data.likelihood.filtered, aes(x=verb.ordered, fill=as.factor(response))) + geom_bar(position="fill") + facet_grid(frame.ordered~.) + scale_x_discrete(name='') + scale_y_continuous(name='Proportion') + scale_fill_brewer(name=element_blank(), type="div") + theme(axis.text.x=element_text(size=12,angle = 45, hjust=1), axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank(), legend.text=element_text(size=15)) 
#dev.off()

##

data.implicature.filtered$exp <- 'implicature'
data.entailment.filtered$exp <- 'entailment'

data.implicature.filtered$strength <- data.implicature.filtered$response != 'Maybe or maybe not'
data.entailment.filtered$strength <- data.entailment.filtered$entailment.polarity!='none'

entailment.implicature <- rbind(subset(data.implicature.filtered, itemtype=='test')[c('subj', 'context', 'item', 'verb', 'frame', 'exp', 'strength')],
                                subset(data.entailment.filtered, itemtype=='test')[c('subj', 'context', 'item', 'verb', 'frame', 'exp', 'strength')])

data.likelihood.filtered.mean <- group_by(data.likelihood.filtered, verb, frame, context, itemtype) %>%
                                 summarise(response.ridit.mean=mean(log(response.ridit)-log(1-response.ridit)))

entailment.implicature <- merge(entailment.implicature, data.likelihood.filtered.mean)

entailment.implicature$verb <- relevel(entailment.implicature$verb, 'hope')
entailment.implicature$frame <- relevel(entailment.implicature$frame, 'NP Ved to VP')

library(MCMCglmm)

logit.prior <- list(R = list(V = 1, nu = 0, fix = 1),
                   G = list(G1 = list(V = diag(29), nu = 29),
                            G2 = list(V=diag(56), nu=56)))

m.strength <- MCMCglmm(strength ~ exp + verb*frame, 
                       random=~idh(1+verb+frame*context):subj + idh(1+verb*frame):context,#subj+context+verb:frame:context,
                       prior = logit.prior,
                       family = "categorical",
                       burnin = 300000,
                       nitt = 1300000,
                       thin = 1000, 
                       pr=T,
                       data = entailment.implicature)

m.strength.like <- MCMCglmm(strength ~ exp*abs(response.ridit.mean), 
                       random=~idh(1+verb+frame*context):subj + idh(1+verb*frame):context,#subj+context+verb:frame:context,
                       prior = logit.prior,
                       family = "categorical",
                       burnin = 300000,
                       nitt = 1300000,
                       thin = 1000, 
                       pr=T,
                       data = entailment.implicature)


m.verb <- MCMCglmm(strength ~ exp*verb*frame, 
                       random=~idh(1+verb+frame*context):subj + idh(1+verb*frame):context,#subj+context+verb:frame:context,
                       prior = logit.prior,
                       family = "categorical",
                       burnin = 300000,
                       nitt = 1300000,
                       thin = 1000, 
                       pr=T,
                       data = entailment.implicature)

data.entailment.filtered$entailed  <- ifelse(data.entailment.filtered$entailment.polarity=='none', F, T)
data.entailment.filtered$direction <- ifelse(data.entailment.filtered$entailment.polarity!='none', data.entailment.filtered$entailment.polarity, NA)

get.verb.frame <- function(verb, frame){
  if (frame == 'NP Ved to VP'){
    if (verb == 'hope'){
      return(m.verb$Sol[,'(Intercept)'])
    } else {
      return(m.verb$Sol[,'(Intercept)']+m.verb$Sol[,paste0('verb', verb)])
    }
    
  } else {
    if (verb == 'hope'){
      return(m.verb$Sol[,'(Intercept)']+m.verb$Sol[,paste0('frame', frame)])
    } else {
      return(m.verb$Sol[,'(Intercept)']+m.verb$Sol[,paste0('verb', verb)]+m.verb$Sol[,paste0('frame', frame)] + m.verb$Sol[,paste0('verb', verb, ':frame', frame)])
    }
  }
}

get.verb.frame <- function(verb, frame){
  return(m.verb[paste0(verb, ':', frame),])
}


compare.verbs <- function(verb1, verb2, frame1, frame2){
  return(mean((get.verb.frame(verb1, frame1)-get.verb.frame(verb2, frame2)) > 0))
}

name.vec <- c()
samp.vec <- c()

for (verb in unique(entailment.implicature$verb)) {
  for (frame in unique(entailment.implicature$frame)) {
    name.vec <- c(name.vec, paste0(verb, ':', frame))
    samp.vec <- rbind(samp.vec, get.verb.frame(verb, frame))
    }
}

row.names(samp.vec) <- name.vec