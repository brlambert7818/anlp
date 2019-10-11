"""
@ author: Brian Lambert
"""

import re
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product
from scipy import optimize


### Q1


def preprocess_line(inline):
    """ Takes line of text and returns string which removes all characters from line not
    in English alphabet, space, digits, or '.'. All characters are lowercased and all
    digits are converted to '0'.

    Args:
       inline (str): single line of input text
    
    Returns:
        text_processed (str): single line of text with unwanted characters removed
    """
    # lowercase all letters
    line = inline.lower()
    # remove replace digits with 0
    line = re.sub('[1-9]', '0', line)
    # remove non-English alphabet, spaces, or .
    return re.sub('[^a-z0.\s#]', '', line)


### Q2


### Q3


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
    """ Returns of dictionary of counts for each unique n-character sequence in a
    give text file.

    Args:
        infile (txt file): Text file to use as training data
        n (int): desired size of n_gram

    Returns: 
        n_counts (dict): dictionary of counts for each n-character sequence
    """
    n_counts = {}
    # with open(infile, 'r') as f:
    # for line in f:
    for line in infile:
        line = preprocess_line(line)
        for j in range(len(line) - (n-1)):
            n_gram = line[j:j + n]
            if n_gram in n_counts:
                n_counts[n_gram] += 1
            else:
                n_counts[n_gram] = 1

    all_combos = product('abcdefghijklmnopqrstuvwxyz0. ', repeat=n)
    if n == 1:
        #all_combos = [a for a in all_combos]
        all_combos = list('abcdefghijklmnopqrstuvwxyz0. ')
    if n == 2:
        all_combos = [a + b for a, b in all_combos]
    if n == 3:
        all_combos = [a + b + c for a, b, c in all_combos]

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
            vocab_size = 29

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
    # creates output file for the model matching the model-br key \tab probability format
    #write_to_file(norm_counts, str(n) + '-gram.txt')
    return norm_counts


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
            sub_key = key[:i+1]
            key_prob += lambdas[i]*ngram_array[i][sub_key]
        model[key] = key_prob
    return model


def perplexity(param, model_type, n, training_data, valid_data):
    """ Perplexity is the measure of how well a probability distribution predicts a sample

    Args:
        param (int or list): int represent the alpha for add-alpha smoothing, list
        representing the lambdas for interpolation smoothing
        model_type (str): Either 'ngram add alpha' or 'interpolation'
        n (int): desired size of n_grams
        training_data (text file): Text file to use as training data
        valid_data (text file): Text file to use as validation or testing data

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


def generate_text(model, n, length, br):
    init_outcomes = np.array(list(model.keys()))
    init_probs = np.array(list(model.values()))
    init_bins = np.cumsum(init_probs)
    init_bins /= np.sum(init_probs)
    text = init_outcomes[np.digitize(np.random.sample(1), init_bins)][0]
    while len(text) < length:
        if text[-1] == '#':
            text += '#'
        condition = text[-(n-1):]
        sub_dict = {key: value for key, value in model.items() if key.startswith(condition)}
        outcomes = np.array(list(sub_dict.keys()))
        probs = np.array(list(sub_dict.values()))
        bins = np.cumsum(probs)
        bins /= np.sum(probs)
        sample = outcomes[np.digitize(np.random.random_sample(), bins)][-1]
        text += sample

    print('Generated Text: ', text)


# reading in modelbr to a probability dict
def input_model(infile):
    model = {}
    data = open(infile).read().splitlines()
    for line in data:
        (key, value) = line.split('\t')
        model[key] = float(value)
    return model


data = open('training.en.txt').read().splitlines()
data_train, data_valid = train_test_split(data, test_size=0.2, shuffle=True)
data_valid, data_test = train_test_split(data_valid, test_size=0.5, shuffle=True)


# sub_dict = {key: value for key, value in trigram.items() if key.startswith('ng') }
# write_to_file(sub_dict, 'ngdict.txt')

# model_br = input_model('model-br.en')
# print(perplexity(data_train, model_br, 3))
# generate_text(model_br, 3, 300, br=True)


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
    optim = optimize.minimize(perplexity,
                              [0.05, 0.2, 0.75],
                              args=optim_args,
                              constraints=({'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] - 1}))
    return optim.x


def optimAddAlpha():
    """ Find the optimal alpha parameter for add-alpha trigram smoothing model

    Returns:
        optim.x (float):  optimized alpha parameter
    """
    model = 'ngram add alpha'
    n = 3
    train = data_train
    valid = data_valid
    optim_args = (model, n, train, valid)
    optim = optimize.minimize(perplexity, 0.07, args=optim_args)
    return float(optim.x)

#model = interpolation(data_train, 3, [0.00270505, 0.27101636, 0.72627859])
#print(perplexity([.05, .25, .7], 'interpolation', 3, data_train, data_valid))
#model = ngram_model(data_train, 3, 0.07)
#write_to_file(model, 'interpolation.txt')

# sub_dict = {key: value for key, value in model.items() if key.startswith('.')}
# count = 0
# for key, value in sub_dict.items():
#     count += value
# print(count)

