import pymc, theano
import numpy as np
import scipy as sp

# import relevant modules
import sys, os, re, argparse, itertools
import theano, pymc
import numpy as np
import scipy as sp

## example: ipython -i -- model.py --loadverbfeatures --loadfeatureloadings --loadjump --loaditem --strengthlevels 5

##################
## argument parser
##################

## initialize parser
parser = argparse.ArgumentParser(description='Load data and run likert factor analysis.')

## file handling
parser.add_argument('--verbs', 
                    type=str, 
                    default='../materials/triad/lists/verbs.list')
parser.add_argument('--data', 
                    type=str, 
                    default='../data/frame/frame.filtered')
parser.add_argument('--output', 
                    type=str, 
                    default='./model')

## general model settings
parser.add_argument('--nonparametric', 
                    nargs='?', 
                    const=True, 
                    default=False)

## feature/category model settings
parser.add_argument('--featureprior',
                    type=str, 
                    choices=['beta', 'dirichlet'], 
                    default='beta')
parser.add_argument('--featuresparsity', 
                    type=float, 
                    default=1.)
parser.add_argument('--strengthprior', 
                    type=str, 
                    choices=['exponential', 'laplace'], 
                    default='exponential')

## inference strength model settings
parser.add_argument('--strengthlevels', 
                    type=int, 
                    default=1)
parser.add_argument('--strengthsparsity', 
                    type=float, 
                    default=1.)

## parameter initialization
parser.add_argument('--loadverbfeatures', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loadstrengthlevels', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loadjump', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loaditem', 
                    nargs='?',
                    const=True,
                    default=False)

## sampler parameters
parser.add_argument('--iterations', 
                    type=int, 
                    default=11000000)
parser.add_argument('--burnin', 
                    type=int, 
                    default=1000000)
parser.add_argument('--thinning', 
                    type=int, 
                    default=10000)

## parse arguments
args = parser.parse_args()

####################
## utility functions
####################

def map_vals_to_indices(col, vals=[]):
    vals = np.unique(col) if not vals else np.array(vals)
    index_mapper = np.vectorize(lambda x: np.where(vals == x))
    
    return vals, index_mapper(col)

############
## load data
############

## read filtered data file
data = np.loadtxt('/home/aaronsteven/experiments/ImplicativeExperiments/results_final.filtered', 
                  delimiter=',', 
                  dtype=np.str,
                  skiprows=1)

## uncomment to look only at matrix positive and matrix negative
#data = data[np.logical_or(data[:,3] == 'negative', data[:,3] == 'positive'), ]

verb_vals = np.array(['manage', 'opt', 'remember', 'think', 'know', 'hasten', 
                      'hesitate', 'mean', 'refuse', 'forget', 'neglect', 'fail', 
                      'want', 'hope'])

subj_vals, subj_indices = map_vals_to_indices(data[:,0])
item_vals, item_indices = map_vals_to_indices(data[:,1])
_, verb_indices = map_vals_to_indices(data[:,2], verb_vals)
negation_vals, negtype_indices = map_vals_to_indices(data[:,3])
response_vals, response_indices = map_vals_to_indices(data[:,4], ['positive', 'soft', 'negative'])
reaction_times = data[:,5]

num_of_subjects = subj_vals.shape[0]
num_of_items = item_vals.shape[0]
num_of_verbs = verb_vals.shape[0]
num_of_negtypes = negation_vals.shape[0]

num_of_observations = data.shape[0]

## feature model

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
#                                                  size=args.strengthlevels-1),
#                          observed=False)

# @pymc.deterministic(trace=False)
# def ibp_stick(feature_prob=feature_prob):
#     probs = np.cumprod(feature_prob)

#     return np.tile(probs, (num_of_verbs-2, num_of_negtypes, 1))

# verb_features = pymc.Bernoulli(name='verb_features',
#                                p=ibp_stick,
#                                value=sp.stats.bernoulli.rvs(.1, size=(num_of_verbs-2,
#                                                                       num_of_negtypes,
#                                                                       args.strengthlevels-1)),
#                                observed=False)

## feature model

if args.featureprior == 'beta':
    feature_prob = pymc.Beta(name='feature_prob',
                             alpha=args.featuresparsity,
                             beta=1.,
                             value=sp.stats.beta.rvs(a=args.featuresparsity,
                                                     b=1., 
                                                     size=args.featurenum),
                             observed=False)

if args.featureprior == 'dirichlet':
    prior_param = np.ones(args.featurenum)*args.featuresparsity
    feature_prob = pymc.Beta(name='feature_prob',
                             theta=prior_param,
                             value=sp.stats.beta.rvs(alpha=prior_param),
                             observed=False)


if args.nonparametric:
    @pymc.deterministic(trace=False)
    def ibp_stick(feature_prob=feature_prob):
        probs = np.cumprod(feature_prob)

        return np.tile(probs, (num_of_verbs,1))


    verb_features = pymc.Bernoulli(name='verb_features',
                                   p=ibp_stick,
                                   value=initialize_verb_features(),
                                   observed=False)

else:

    @pymc.deterministic(trace=False)
    def feature_prob_tile(probs=feature_prob):
        return np.tile(probs, (num_of_verbs,1))

    
    verb_features = pymc.Bernoulli(name='verb_features',
                                   p=feature_prob_tile,
                                   value=initialize_verb_features(),
                                   observed=False)


feature_prob = pymc.Beta(name='feature_prob',
                         alpha=args.featuresparsity,
                         beta=1.,
                         value=sp.stats.beta.rvs(a=args.featuresparsity,
                                                 b=1., 
                                                 size=args.strengthlevels-1),
                         observed=False)

@pymc.deterministic(trace=False)
def feature_prob_tile(probs=feature_prob):
    return np.tile(probs, (num_of_verbs-2, num_of_negtypes, 1))


verb_features = pymc.Bernoulli(name='verb_features',
                               p=feature_prob_tile,
                               value=sp.stats.bernoulli.rvs(.5, size=(num_of_verbs-2,
                                                                      num_of_negtypes,
                                                                      args.strengthlevels-1)),
                               observed=False)

@pymc.deterministic
def verb_features_with_controls_and_intercept(verb_features=verb_features):
    verb_features_with_controls = np.append(verb_features, np.zeros([2, num_of_negtypes, args.strengthlevels-1]), axis=0)
    
    return np.append(np.ones([num_of_verbs, num_of_negtypes, 1]), verb_features_with_controls, axis=2)


## strength model

strength = pymc.Exponential(name='strength',
                            beta=1.,
                            value=sp.stats.expon.rvs(scale=1.,
                                                     size=args.strengthlevels),
                            observed=False)

## polarity model

polarity = pymc.Bernoulli(name='polarity',
                          p=.5,
                          value=sp.stats.bernoulli.rvs(.5, size=(num_of_verbs, 
                                                                 num_of_negtypes)),
                          observed=False)

#################
## random effects
#################

## subject bias model

subj_bias_prior = pymc.Exponential(name='subj_bias_prior',
                                   beta=1.,
                                   value=sp.stats.expon.rvs(scale=1., size=num_of_negtypes),
                                   observed=False)

# @pymc.deterministic(trace=False)
# def subj_bias_prior_tile(prior=subj_bias_prior):
#     return np.tile(prior, (num_of_subjects, 1))

subj_bias = pymc.Normal(name='subj_bias',
                        mu=0.,
                        tau=subj_bias_prior,
                        value=np.random.normal(0., 
                                               1.,
                                               size=(num_of_subjects, 
                                                     num_of_negtypes)),
                        observed=False)

## item bias model

item_bias_prior = pymc.Exponential(name='item_bias_prior',
                                   beta=1.,
                                   value=sp.stats.expon.rvs(scale=1., size=num_of_negtypes),
                                   observed=False)

# @pymc.deterministic(trace=False)
# def item_bias_prior_tile(prior=item_bias_prior):
#     return np.tile(prior, (num_of_items, 1))

item_bias = pymc.Normal(name='item_bias',
                        mu=0.,
                        tau=item_bias_prior,
                        value=np.random.normal(0., 
                                               1.,
                                               size=(num_of_items, 
                                                     num_of_negtypes)),
                        observed=False)


## subject error model

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
                            value=np.ones(2),
                            observed=False)

@pymc.deterministic
def entailment(features=verb_features_with_controls_and_intercept, strength=strength, polarity=polarity):
    return np.where(polarity == 1, 1, -1) * np.dot(features, strength)

@pymc.deterministic
def response_prob(entailment=entailment, subj_bias=subj_bias, item_bias=item_bias, response_error=response_error, cutpoint=cutpoint):
    bias = subj_bias[subj_indices, negtype_indices] + item_bias[item_indices, negtype_indices]
    direction =  np.where(response_error == 1, -1, 1)

    perceived_entailment = direction * entailment[verb_indices, negtype_indices] / bias

    cumprobs = pymc.invlogit(np.array([perceived_entailment-cutpoint[0], perceived_entailment+cutpoint[1]]).transpose())

    zeros = np.zeros(num_of_observations)[:, None]
    ones = np.ones(num_of_observations)[:, None]

    return np.append(cumprobs, ones, axis=1) - np.append(zeros, cumprobs, axis=1)

## observation model

observations = pymc.Categorical(name='observations',
                                p=response_prob,
                                value=response_indices,
                                observed=True)


############
## fit model
############

## initialize model and begin sampler
model = pymc.MCMC(locals())
model.sample(iter=args.iterations, burn=args.burnin, thin=args.thinning)

## get deviance trace, minimum deviance, and index of minimum deviance
deviance_trace = model.trace('deviance')()
deviance_min = deviance_trace.min()
minimum_index = np.where(deviance_trace == deviance_min)[0][0]

## get best fixed effects
verb_features_best = model.verb_features.trace()[minimum_index]
feature_loadings_best = model.feature_loadings.trace()[minimum_index]

## get best random effects
jump_best = model.jump.trace()[minimum_index]
item_best = model.item.trace()[minimum_index]
