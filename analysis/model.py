import pymc, theano
import numpy as np
import scipy as sp

##
## functions
##

def map_vals_to_indices(col, vals=[]):
    vals = np.unique(col) if not list(vals) else np.array(vals)
    index_mapper = np.vectorize(lambda x: np.where(vals == x))

    return vals, index_mapper(col)

## load implicatives data

data = np.loadtxt('/home/aaronsteven/experiments/ImplicativeExperiments/results_final.filtered', 
                  delimiter=',', 
                  dtype=np.str,
                  skiprows=1)

#data = data[np.logical_and(data[:,2] != 'want', data[:,2] != 'hope'), ]
data = data[np.logical_or(data[:,3] == 'negative', data[:,3] == 'positive'), ]

verb_vals = np.array(['fail', 'forget', 'neglect', 'refuse', 'manage', 'remember', 'opt', 'think', 'know', 'hasten', 'hesitate', 'mean', 'want', 'hope'])

# subj,item,verb,matrix,followup,answer,answerRT

subj_vals, subj_indices = map_vals_to_indices(data[:,0])
item_vals, item_indices = map_vals_to_indices(data[:,1])
_, verb_indices = map_vals_to_indices(data[:,2], verb_vals)
negation_vals, negtype_indices = map_vals_to_indices(data[:,3])
response_vals, response_indices = map_vals_to_indices(data[:,4], ['positive', 'soft', 'negative'])

#data = data.astype(np.int)

num_of_subjects = subj_vals.shape[0]
num_of_items = item_vals.shape[0]
num_of_verbs = verb_vals.shape[0]
num_of_negtypes = negation_vals.shape[0]

num_of_observations = data.shape[0]

## feature model

feature_sparsity = 0.5
num_of_features = 7

# sparsity_hyperprior = 2.

# feature_sparsity = pymc.Exponential(name='feature_sparsity',
#                                     beta=sparsity_hyperprior,
#                                     value=sp.stats.expon.rvs(scale=sparsity_hyperprior),
#                                     observed=False)


# feature_prob = pymc.Beta(name='feature_prob',
#                          alpha=feature_sparsity,
#                          beta=1.,
#                          value=sp.stats.beta.rvs(a=feature_sparsity,#1./sparsity_hyperprior,
#                                                  b=1., 
#                                                  size=num_of_features-1),
#                          observed=False)

# @pymc.deterministic(trace=False)
# def ibp_stick(feature_prob=feature_prob):
#     probs = np.cumprod(feature_prob)

#     return np.tile(probs, (num_of_verbs-2, num_of_negtypes, 1))

# verb_features = pymc.Bernoulli(name='verb_features',
#                                p=ibp_stick,
#                                value=sp.stats.bernoulli.rvs(.1, size=(num_of_verbs-2,
#                                                                       num_of_negtypes,
#                                                                       num_of_features-1)),
#                                observed=False)

feature_prob = pymc.Beta(name='feature_prob',
                         alpha=1.,
                         beta=1.,
                         value=sp.stats.beta.rvs(a=1.,#1./sparsity_hyperprior,
                                                 b=1., 
                                                 size=num_of_features-1),
                         observed=False)

@pymc.deterministic(trace=False)
def feature_prob_tile(probs=feature_prob):
    return np.tile(probs, (num_of_verbs-2, num_of_negtypes, 1))


verb_features = pymc.Bernoulli(name='verb_features',
                               p=feature_prob_tile,
                               value=sp.stats.bernoulli.rvs(.5, size=(num_of_verbs-2,
                                                                      num_of_negtypes,
                                                                      num_of_features-1)),
                               observed=False)

@pymc.deterministic
def verb_features_with_controls_and_intercept(verb_features=verb_features):
    verb_features_with_controls = np.append(verb_features, np.zeros([2, num_of_negtypes, num_of_features-1]), axis=0)
    
    return np.append(np.ones([num_of_verbs, num_of_negtypes, 1]), verb_features_with_controls, axis=2)


## strength model

# strength_prior = pymc.Exponential(name='strength_prior',
#                               beta=1.,
#                               value=sp.stats.expon.rvs(scale=1.),
#                               observed=False)

strength = pymc.Exponential(name='strength',
                            beta=1.,
                            value=sp.stats.expon.rvs(scale=1.,
                                                     size=num_of_features),
                            observed=False)

## concordance model

# reverse_prior = pymc.Beta(name='reverse_prior',
#                           alpha=1.,
#                           beta=1.,
#                           value=.5,
#                           observed=False)

polarity = pymc.Bernoulli(name='polarity',
                          p=.5,
                          value=sp.stats.bernoulli.rvs(.5, size=(num_of_verbs, 
                                                                 num_of_negtypes)),
                          observed=False)

## bias model

## subject

subj_bias_prior = pymc.Exponential(name='subj_bias_prior',
                                   beta=1.,
                                   value=sp.stats.expon.rvs(scale=1., size=num_of_negtypes),
                                   observed=False)

@pymc.deterministic(trace=False)
def subj_bias_prior_tile(prior=subj_bias_prior):
    return np.tile(prior, (num_of_subjects, 1))

subj_bias = pymc.Exponential(name='subj_bias',
                             beta=subj_bias_prior_tile,
                             value=sp.stats.expon.rvs(scale=1.,
                                                      size=(num_of_subjects, 
                                                            num_of_negtypes)),
                             observed=False)

## item

item_bias_prior = pymc.Exponential(name='item_bias_prior',
                                   beta=1.,
                                   value=sp.stats.expon.rvs(scale=1., size=num_of_negtypes),
                                   observed=False)

@pymc.deterministic(trace=False)
def item_bias_prior_tile(prior=item_bias_prior):
    return np.tile(prior, (num_of_items, 1))


item_bias = pymc.Exponential(name='item_bias',
                             beta=item_bias_prior_tile,
                             value=sp.stats.expon.rvs(scale=1.,
                                                      size=(num_of_items, 
                                                            num_of_negtypes)),
                             observed=False)

## error model

subj_error_prior = pymc.Exponential(name='subj_error_prior',
                                   beta=1.,
                                   value=sp.stats.expon.rvs(scale=1., size=num_of_negtypes),
                                   observed=False)

@pymc.deterministic(trace=False)
def subj_error_prior_tile(prior=subj_error_prior):
    return np.tile(prior, (num_of_subjects, 1))

subj_error = pymc.Beta(name='subj_error',
                       alpha=subj_error_prior_tile,
                       beta=np.ones([num_of_subjects, num_of_negtypes]),
                       value=sp.stats.beta.rvs(a=0.1,
                                               b=1., 
                                               size=(num_of_subjects,
                                                     num_of_negtypes)),
                       observed=False)

response_error = pymc.Bernoulli(name='response_error',
                                p=subj_error[subj_indices, negtype_indices],
                                value=sp.stats.bernoulli.rvs(.01, size=num_of_observations),
                                observed=False)

## response model

cutpoint = pymc.Exponential(name='cutpoints',
                            beta=1.,
                            value=np.ones([num_of_negtypes, 2]),
                            observed=False)

@pymc.deterministic
def entailment(features=verb_features_with_controls_and_intercept, strength=strength, polarity=polarity):
    return np.where(polarity == 1, 1, -1) * np.dot(features, strength)

@pymc.deterministic(trace=False)
def response_prob(entailment=entailment, subj_bias=subj_bias, item_bias=item_bias, response_error=response_error, cutpoint=cutpoint):
    bias = subj_bias[subj_indices, negtype_indices] + item_bias[item_indices, negtype_indices]
    direction =  np.where(response_error == 1, -1, 1)

    perceived_entailment = direction * entailment[verb_indices, negtype_indices] / bias

    cumprobs = pymc.invlogit(np.array([perceived_entailment-cutpoint[negtype_indices, 0], perceived_entailment+cutpoint[negtype_indices, 1]]).transpose())

    zeros = np.zeros(num_of_observations)[:, None]
    ones = np.ones(num_of_observations)[:, None]

    return np.append(cumprobs, ones, axis=1) - np.append(zeros, cumprobs, axis=1)

## observation model

observations = pymc.Categorical(name='observations',
                                p=response_prob,
                                value=response_indices,
                                observed=True)

## sampler

model = pymc.MCMC(locals())
model.sample(iter=5000000, burn=1000000, thin=1000)
