'''
@ author: Brian Lambert, Jakub Onysk
'''

import re
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product
from scipy import optimize


### Q1 ###

def preprocess_line(line):
    """ Takes line of text and returns string which removes all characters from line not
    in English alphabet, space, digits, '#' or '.'. All characters are lowercased and all
    digits are converted to '0'. Each line is also encased in '#' characters. 
    Args:
       inline (str): single line of input text
    
    Returns:
        text_processed (str): single line of text with unwanted characters removed and
        encased in #'s
    """
    # lowercase all letters
    line = line.lower()
    # remove replace digits with 0
    line = re.sub('[1-9]', '0', line)
    #add #'s around a sequence
    if (not line.startswith('#')) and (not line.endswith('#')):
        line = '#'+line+'#'
    # remove non-English alphabet, spaces, #, or .
    return re.sub('[^a-z0.\s#]', '', line)


### Q3 ###

def write_to_file(model, file_name):
    """ Writes a dictionary to a text file with each line containing a key and its value
    Args:
        model (dict): language model with n-gram as key and its probability as value
        file_name (str): name of output file
    Returns:
        None: creates file in current directory
    """
    with open(file_name, 'w') as f:
        dict_sorted = dict(sorted(model.items()))
        for key, value in dict_sorted.items():
            f.write(key + '    ' + str(value))
            f.write('\n')


def generate_counts(infile, n):
    """ Returns a dictionary of counts for each unique n-character string in a given 
    text file and supplements it with missing strings and assigns count 0.
    Args:
        infile (txt file): Text file to use as training data
        n (int): desired size of n_gram
    Returns: 
        n_counts (dict): dictionary of counts for each n-character string
    """
    n_counts = {}
    for line in infile:
        line = preprocess_line(line)
        for j in range(len(line) - (n-1)):
            n_gram = line[j:j + n]
            if n_gram in n_counts:
                n_counts[n_gram] += 1
            else:
                n_counts[n_gram] = 1

    all_combos = product('#abcdefghijklmnopqrstuvwxyz0. ', repeat=n)
    if n == 1:
#        all_combos = [a for a in all_combos]
        all_combos = list('#abcdefghijklmnopqrstuvwxyz0. ')
    if n == 2:
        all_combos = [a + b for a, b in all_combos]
        regex = re.compile(r'##')
        all_combos = [i for i in all_combos if not regex.match(i)]
    if n == 3:
        all_combos = [a + b + c for a, b, c in all_combos]
        regex = re.compile(r'.#.')
        all_combos = [i for i in all_combos if not regex.match(i)]

    all_combo_dict = dict.fromkeys(all_combos, 0)
    all_combo_dict.update(n_counts)
    return all_combo_dict

def ngram_model(infile, n, add_alpha=None):
    """ Creates an n-character language model using a training data set and outputs
    a dictionary of the model probabilities.   
    Args:
        infile (text file): Text file to use as training data 
        n (int): desired size of n_grams
        add_alpha (float): optional function parameter to implement add-alpha smoothing.
    
    Returns:
        norm_counts (dict): all n-character sequences with estimated normalized
        probabilities 
    """
    if n < 1:
        raise Exception('n must be >= 1')

    # if add_alpha is not included in the function call then alpha and vocab_size
    # will remain zero and a basic n-gram model will be generated
    alpha = vocab_size = 0
    if add_alpha is not None:
        if add_alpha > 1:
            raise Exception('Alpha must be <= 1')
        else:
            alpha = add_alpha
            vocab_size = 30

    norm_counts = {}
    n_counts = generate_counts(infile, n)
    n_minus_counts = generate_counts(infile, n-1)
    
    # find the counts for the n-1 gram model to use for the divisor in
    # probability calculation
    for key, value in n_counts.items():
        n_minus_key = key[:n - 1]
        if n_minus_counts[n_minus_key] != 0 or add_alpha is not None:
            norm_counts[key] = (n_counts[key] + alpha) \
                               / (n_minus_counts[n_minus_key] + alpha * vocab_size)
        else:
            norm_counts[key] = 0

    return norm_counts

def perplexity_for_optim(param, model_type, n, training_data, valid_data):
    """ Perplexity is the measure of how well a probability distribution predicts a sample.
        This variation of a function is used to optimise alpha for add-alpha smoothing.
    Args:
        param (int or list): int represent the alpha for add-alpha smoothing, list
        representing the lambdas for interpolation smoothing
        model_type (str): Either 'ngram add alpha' or 'interpolation'
        n (int): desired size of n_grams
        training_data (list): Training data set
        valid_data (list): Validation data set
    Returns:
        (float): perplexity calculation for given model
    """
    model = None
    if model_type == 'ngram add alpha':
        model = ngram_model(training_data, n, param)
    elif model_type == 'interpolation':
        model = interpolation(training_data, n, param)

    p_log = 0
    n_gram_count = 0
    for line in valid_data:
        line = preprocess_line(line)
        for j in range(len(line) - (n-1)):
            n_gram = line[j:j + n]
            n_gram_count += 1
            p_log += np.log2(model[n_gram])
    return 2 ** (-1 / n_gram_count * p_log)

def optimAddAlpha(train_set, valid_set, n):
    """ Find the optimal alpha parameter for add-alpha trigram smoothing model
    Returns:
        optim.x (float):  optimized alpha parameter
    """
    model = 'ngram add alpha'
    train = train_set
    valid = valid_set
    optim_args = (model, n, train, valid)
    bnds=((0,1),)
    optim = optimize.minimize(perplexity_for_optim, 0.04,bounds=bnds, args=optim_args)
    return float(optim.x)                  

def perplexity(infile, n_gram_model, n):
    """ Perplexity is the measure of how well a probability distribution predicts a sample.
        This is a simple perplexity function that just calculates perplexity of a document
        using a given model, without optimising any parameters.
    Args:
        infile (list): a set of sequences to calculate perplexity on
        n_gram_model (dict): an n-gram model to use in perplexity calculation
        n (int): desired size of n_grams
    Returns:
        (float): perplexity calculation for given model
    """
    p_log = 0
    n_gram_count = 0
    for line in infile:
        line = preprocess_line(line)
        for j in range(len(line) - (n-1)):
            n_gram = line[j:j + n]
            n_gram_count += 1
            p_log += np.log2(n_gram_model[n_gram])
    return 2 ** (-1 / n_gram_count * p_log)


## English data ##
data = open('training.en').read().splitlines()
data_train, data_valid = train_test_split(data, test_size=0.2, shuffle=True)
data_valid, data_test = train_test_split(data_valid, test_size=0.5, shuffle=True)

# Find best alpha for English #
best_alpha=optimAddAlpha(data_train, data_valid, 3)

# Trigram model for English + save to file #
trigram = ngram_model(data_train, 3, add_alpha=best_alpha)
write_to_file(trigram, 'trigram.txt')

# Find two-character history 'ng' #
ng_ngrams = {key: value for key, value in trigram.items() if key.startswith('ng') }
write_to_file(ng_ngrams, 'ngdict.txt')

#Perplexities on validation and test set
print(perplexity(data_train, trigram, 3)) # Perplexity of english model on the training data
print(perplexity(data_valid, trigram, 3)) # Perplexity of english model on the validation data
print(perplexity(data_test, trigram, 3)) # Perplexity of english model on the test data


## German data ##
dataDe = open('training.de').read().splitlines()
data_trainDe, data_validDe = train_test_split(dataDe, test_size=0.2, shuffle=True)
data_validDe, data_testDe = train_test_split(data_validDe, test_size=0.5, shuffle=True)

# Find best alpha for German #
best_alphaDe = optimAddAlpha(data_trainDe, data_validDe, 3)

# Trigram model for German
trigramDe = ngram_model(data_trainDe, 3, add_alpha=best_alphaDe)

# Perplexities on training, validation and test set for German
print(perplexity(data_trainDe, trigramDe, 3))
print(perplexity(data_validDe, trigramDe, 3))
print(perplexity(data_testDe, trigramDe, 3))


## Spanish data ##
dataEs = open('training.es').read().splitlines()
data_trainEs, data_validEs = train_test_split(dataEs, test_size=0.2, shuffle=True)
data_validEs, data_testEs = train_test_split(data_validEs, test_size=0.5, shuffle=True)

# Find best alpha for Spanish #
best_alphaEs = optimAddAlpha(data_trainEs, data_validEs, 3)

# Trigram model for Spanish #
trigramEs = ngram_model(data_trainEs, 3, add_alpha=best_alphaEs)

# Perplexities on training, validation and test set for Spanish
print(perplexity(data_trainEs, trigramEs, 3))
print(perplexity(data_validEs, trigramEs, 3))
print(perplexity(data_testEs, trigramEs, 3))


### Q4 ###

def generate_text(model, n, length):
    """ Generates a 'length'-character text based on a given n-gram model.
    
    Args:
        model (dict): From what model to generate text
        n (int): desired size of n-grams
        length (int): desired length of a generated text
        
    Returns:
        text (str): Desired generated text of length 'length' 
    
    """
    init_model = {key: value for key, value in model.items() if key.startswith('#') and not key.endswith('#')}
    init_outcomes = np.array(list(init_model.keys()))
    init_probs = np.array(list(init_model.values()))
    init_bins = np.cumsum(init_probs)/np.sum(init_probs)
    text = init_outcomes[np.digitize(np.random.sample(1), init_bins)][0]
    while len(text) < length:
        if text[-1] == '#':
            text += init_outcomes[np.digitize(np.random.sample(1), init_bins)][0]
        condition = text[-(n-1):]
        sub_dict = {key: value for key, value in model.items() if key.startswith(condition)}
        outcomes = np.array(list(sub_dict.keys()))
        probs = np.array(list(sub_dict.values()))
        bins = np.cumsum(probs)/np.sum(probs)
        sample = outcomes[np.digitize(np.random.random_sample(), bins)][-1]
        text += sample

    print('Generated Text: ', text)

def input_model(infile):
    """ Loads in a pre-trained model file and converts it into a dictionary based model.
    
    Args:
        infile (text file): Text file with a pre-trained model.
    
    Returns:
        model (dict): Model with strings and their corresponding probabilities.
    """
    model = {}
    data = open(infile).read().splitlines()
    for line in data:
        (key, value) = line.split('\t')
        model[key] = float(value)
    return model

## Generate from trigram model ##
generate_text(trigram, 3, 300)

## Generate from model-br ##
model_br = input_model('model-br.en')
generate_text(model_br, 3, 300)


### Q5 ###

## Test document model and perplexity on trigram and model-br ##   
testData = open('test').read().splitlines()
print(perplexity(testData, trigram, 3)) # English trigram model
print(perplexity(testData, model_br, 3)) # model-br trigram model

                  
## Perplexities on diff. lang models ##                  
print(perplexity(testData, trigram, 3)) # English model perplexity 
print(perplexity(testData, trigramDe, 3)) # German model perplexity
print(perplexity(testData, trigramEs, 3)) # Spanish model perplexity                  
                  

## Warmup model perplexity check
warmupModel = {'##a':0.2, '#aa':0.2, '#ba':0.15, 'aaa':0.4, 'aba':0.6, 
               'baa':0.25, 'bba':0.5, '##b': 0.8, '#ab':0.7, '#bb':0.75, 
               'aab':0.5, 'abb':0.3, 'bab':0.65, 'bbb':0.4, '###':0.0,
               '#a#':0.1, '#b#':0.1, 'aa#':0.1, 'ab#':0.1, 'ba#':0.1,'bb#':0.1}
               
print(perplexity(['##abaab#'],warmupModel,3))                  
                  
                  
### Q6 ### 

def interpolation(data_train, n, lambdas):
    """ Creates an n-gram language model using interpolation smoothing
    Args:
        data_train (text file): Text file to use as training data
        n (int): desired size of n_grams
        lambdas (list): list of weights of length n to be weight the probability of each
                        n-gram model. List order should be (1, 2, ..., n-1, n) for
                        n-gram size = n
    Returns:
        model (dict): all n-character sequences with estimated probabilities
    """
    ngram_array = [dict() for i in range(n)]
    for i in range(n):
        ngram_array[i] = ngram_model(data_train, i + 1)
    model = {}
    for key, value in ngram_array[-1].items():
        key_prob = 0
        for i in range(n):
            sub_key = key[-(i+1):]
            key_prob += lambdas[i]*ngram_array[i][sub_key]
        model[key] = key_prob
    return model   

def optimInterpolation():
    """ Find the optimal lambda parameters for interpolation trigram smoothing model
    Returns:
        optim.x (ndarray): array of optimized lambda parameters
    """
    model = 'interpolation'
    n = 3
    train = data_train
    valid = data_valid
    optim_args = (model, n, train, valid)
    optim = optimize.minimize(perplexity_for_optim,
                              [0.05, 0.2, 0.75],
                              args=optim_args,
                              constraints=({'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] - 1}))
    return optim.x

## Implement interpolation ##
lambdasInter = optimInterpolation()
modelInter = interpolation(data_train, 3, lambdasInter)    
              
# Perplexities from interpolation method #
print(perplexity(data_train, modelInter, 3))
print(perplexity(data_valid, modelInter, 3))
print(perplexity(data_test, modelInter, 3))
print(perplexity(testData, modelInter, 3))
              
# Generate text using interpolation method #
generate_text(modelInter, 3, 300)

