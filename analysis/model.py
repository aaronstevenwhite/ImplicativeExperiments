import sys, os, re, argparse, itertools
import theano, pymc
import numpy as np
import scipy as sp

## example: ipython -i -- model.py --loadverbfeatures --loadfeatureloadings --loadjump --loaditem --strengthlevels 5

##################
## argument parser
##################

## initialize parser
parser = argparse.ArgumentParser(description='Load data and run implicatives model.')

## file handling
parser.add_argument('--data', 
                    type=str, 
                    default='../data/results.filtered')
parser.add_argument('--output', 
                    type=str, 
                    default='./model')

## model structure settings
parser.add_argument('--nonparametric', 
                    nargs='?', 
                    const=True, 
                    default=False)
parser.add_argument('--featureprior',
                    type=str, 
                    choices=['beta', 'dirichlet'], 
                    default='beta')
parser.add_argument('--featurebynegtype', 
                    nargs='?', 
                    const=True, 
                    default=False)
parser.add_argument('--cutpoints',
                    type=str, 
                    choices=['equidistant', 'direction', 'negtype', 'direction+negtype'], 
                    default='equidistant')


## model hyperparameters
parser.add_argument('--featuresparsity', 
                    type=float, 
                    default=1.)
parser.add_argument('--strengthprior', 
                    type=str, 
                    choices=['exponential', 'laplace'], 
                    default='exponential')
parser.add_argument('--strengthlevels', 
                    type=int, 
                    default=2)
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
parser.add_argument('--loadsubjbias', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loaditembias', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loadresponseerror', 
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

## raise error for unimplemented combinations
if args.featureprior == 'dirichlet' and args.featurebynegtype:
    ## the current issue with this model is that pymc.Multinomial cannot take
    ## values that are tensors of order greater than 2 (matrices)
    raise NotImplementedError, 'this dirichlet model is not yet implemented'

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
data = np.loadtxt(args.data, 
                  delimiter=',', 
                  dtype=np.str,
                  skiprows=1)

## uncomment to look only at matrix positive and matrix negative
#data = data[np.logical_or(data[:,3] == 'negative', data[:,3] == 'positive'), ]

## set verb order (this is important to ensure "want" and "hope" are treated as baselines)
verb_vals = ['manage', 'opt', 'remember', 'think', 'know', 'hasten', 
             'hesitate', 'mean', 'refuse', 'forget', 'neglect', 'fail',
             'want', 'hope']
baseline_verb_vals = np.array(['want', 'hope'])

## extract data
subj_vals, subj_indices = map_vals_to_indices(data[:,0])
item_vals, item_indices = map_vals_to_indices(data[:,1])
verb_vals, verb_indices = map_vals_to_indices(data[:,2], verb_vals)
negation_vals, negtype_indices = map_vals_to_indices(data[:,3])
response_vals, response_indices = map_vals_to_indices(data[:,4], ['positive', 'soft', 'negative'])
reaction_times = data[:,5]

## set constants
num_of_subjects = subj_vals.shape[0]
num_of_items = item_vals.shape[0]
num_of_verbs = verb_vals.shape[0]
num_of_baseline_verbs = baseline_verb_vals.shape[0]
num_of_negtypes = negation_vals.shape[0]
num_of_observations = data.shape[0]

#################
## initialization
#################

def get_verb_features_shape(initial_dim=num_of_verbs-num_of_baseline_verbs, 
                            final_dim=args.strengthlevels-1):
    if args.featurebynegtype:
        return (initial_dim, num_of_negtypes, final_dim)
    else:
        return (initial_dim, final_dim)

def get_strength_shape():
    if args.featurebynegtype:
        return args.strengthlevels
    else:
        return (args.strengthlevels, num_of_negtypes)


def initialize_feature_prior():
    if args.featureprior == 'beta':
        return sp.stats.beta.rvs(a=args.featuresparsity,
                                 b=1., 
                                 size=args.strengthlevels-1)

    elif args.featureprior == 'dirichlet':
        if args.nonparametric:
            return sp.stats.beta.rvs(a=1.,
                                     b=args.featuresparsity,
                                     size=args.strengthlevels-1)
        else:
            return sp.stats.dirichlet(args.featuresparsity*np.ones(args.strengthlevels-1))

def initialize_verb_features():
    if args.loadverbfeatures:
        raise NotImplementedError
    else:
        return sp.stats.bernoulli.rvs(.5, size=get_verb_features_shape())

################
## feature model
################

if args.featureprior == 'beta':
    raw_prob = pymc.Beta(name='raw_prob',
                         alpha=args.featuresparsity,
                         beta=1.,
                         value=initialize_feature_prior(),
                         observed=False)

    if args.nonparametric:
        @pymc.deterministic(trace=False)
        def feature_prob(raw_probs=raw_prob):
            return np.tile(np.cumprod(raw_probs), 
                           get_verb_features_shape(final_dim=1))

    else:
        @pymc.deterministic(trace=False)
        def feature_prob(raw_probs=raw_prob):
            return np.tile(raw_probs, 
                           get_verb_features_shape(final_dim=1))

    verb_features = pymc.Bernoulli(name='verb_features',
                                   p=feature_prob,
                                   value=initialize_verb_features(),
                                   observed=False)


elif args.featureprior == 'dirichlet':

    if args.nonparametric:
        raw_prob = pymc.Beta(name='raw_prob',
                                 alpha=1.,
                                 beta=args.featuresparsity,
                                 value=initialize_feature_prior(),
                                 observed=False)

        @pymc.deterministic(trace=False)
        def feature_prob(raw_probs=raw_prob):
            raw_probs_flipped = np.append(1., np.cumprod(1-raw_probs))

            return np.tile(raw_probs*raw_probs_flipped, 
                           get_verb_features_shape(final_dim=1))

    else:
        dir_hyperparameter = args.featuresparsity*np.ones(args.strengthlevels)
        raw_prob = pymc.Dirichlet(name='feature_prob',
                                      theta=dir_hyperparameter,
                                      value=initialize_feature_prior(),
                                      observed=False)

        @pymc.deterministic(trace=False)
        def feature_prob(raw_probs=raw_prob):
            return np.tile(raw_probs, 
                           get_verb_features_shape(final_dim=1))    


        verb_features = pymc.Multinomial(name='verb_features',
                                         n=1,
                                         p=feature_prob,
                                         value=initialize_verb_features(),
                                         observed=False)


@pymc.deterministic
def verb_features_with_controls_and_intercept(verb_features=verb_features):
    control_shape = get_verb_features_shape(initial_dim=num_of_baseline_verbs)
    verb_features_with_controls = np.append(verb_features, 
                                            np.zeros(control_shape), 
                                            axis=0)

    return np.append(np.ones(get_verb_features_shape(initial_dim=num_of_verbs, 
                                                     final_dim=1)),
                     verb_features_with_controls, 
                     axis=len(control_shape)-1)


##########################
## strength/polarity model
##########################

if args.strengthprior=='exponential':
    strength = pymc.Exponential(name='strength',
                                beta=1.,
                                value=sp.stats.expon.rvs(scale=1.,
                                                         size=get_strength_shape()),
                                observed=False)

    polarity = pymc.Bernoulli(name='polarity',
                              p=.5,
                              value=sp.stats.bernoulli.rvs(.5, size=(num_of_verbs, 
                                                                     num_of_negtypes)),
                              observed=False)

    @pymc.deterministic
    def entailment(features=verb_features_with_controls_and_intercept, strength=strength, polarity=polarity):
        return np.where(polarity == 1, 1, -1) * np.dot(features, strength)


elif args.strengthprior=='laplace':
    ## pymc has a laplace distribution but its matrix form is broken
    ## the following formulation involving exponentials is equivalent
    ## (see wikipedia)
    strength1 = pymc.Exponential(name='strength1',
                                beta=1.,
                                value=sp.stats.expon.rvs(scale=1.,
                                                         size=get_strength_shape()),
                                observed=False)

    strength2 = pymc.Exponential(name='strength2',
                                beta=1.,
                                value=sp.stats.expon.rvs(scale=1.,
                                                         size=get_strength_shape()),
                                observed=False)

    @pymc.deterministic
    def strength(strength1=strength1, strength2=strength2):
        return strength1 - strength2

    @pymc.deterministic
    def entailment(features=verb_features_with_controls_and_intercept, strength=strength):
        return np.dot(features, strength)


#################
## random effects
#################

## subject bias model

subj_bias_prior = pymc.Exponential(name='subj_bias_prior',
                                   beta=1.,
                                   value=sp.stats.expon.rvs(scale=1., size=num_of_negtypes),
                                   observed=False)


@pymc.deterministic(trace=False)
def subj_bias_prior_tile(prior=subj_bias_prior):
    return np.tile(prior, (num_of_subjects, 1))

subj_bias = pymc.Normal(name='subj_bias',
                        mu=0.,
                        tau=subj_bias_prior_tile,
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

@pymc.deterministic(trace=False)
def item_bias_prior_tile(prior=item_bias_prior):
    return np.tile(prior, (num_of_items, 1))

item_bias = pymc.Normal(name='item_bias',
                        mu=0.,
                        tau=item_bias_prior_tile,
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

if args.cutpoints == 'equidistant':
    cutpoint_val = pymc.Exponential(name='cutpoint_val',
                                    beta=1.,
                                    value=1.,
                                    observed=False)

    @pymc.deterministic
    def cutpoints(cutpoint_val=cutpoint_val):
        return np.tile(cutpoint_val, [2, num_of_negtypes])

elif args.cutpoints == 'direction':
    cutpoint_val = pymc.Exponential(name='cutpoint_val',
                                beta=1.,
                                value=np.ones(2),
                                observed=False)

    @pymc.deterministic
    def cutpoints(cutpoint_val=cutpoint_val):
        return np.tile(cutpoint_val, [1, num_of_negtypes])

elif args.cutpoints == 'negtype':
    cutpoint_val = pymc.Exponential(name='cutpoint_val',
                                beta=1.,
                                value=np.ones(num_of_negtypes),
                                observed=False)

    @pymc.deterministic
    def cutpoints(cutpoint_val=cutpoint_val):
        return np.tile(cutpoint_val, [2, 1])

elif args.cutpoints == 'direction+negtype':
    cutpoints = pymc.Exponential(name='cutpoints',
                                 beta=1.,
                                 value=np.ones([2, num_of_negtypes]),
                                 observed=False)

@pymc.deterministic
def response_prob(entailment=entailment, subj_bias=subj_bias, item_bias=item_bias, response_error=response_error, cutpoints=cutpoints):
    bias = subj_bias[subj_indices, negtype_indices] + item_bias[item_indices, negtype_indices]
    direction =  np.where(response_error == 1, -1, 1)

    perceived_entailment = direction * entailment[verb_indices, negtype_indices] / bias

    cumprobs = pymc.invlogit(np.array([perceived_entailment-cutpoints[0, negtype_indices], 
                                       perceived_entailment+cutpoints[1, negtype_indices]]).transpose())

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
