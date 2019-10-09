'''
@ author: Brian Lambert
'''

import re
import os
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product


### Q1

def preprocess_line(line):
    '''
    Takes line of text and returns string which removes all characters from line not
    in English alphabet, space, digits, or '.'. All characters are lowercased and all
    digits are converted to '0'.

    Parameters:
        text (str): single line of input text
    
    Returns:
        text_processed (str): single line of text with unwanted characters removed
    '''
    # lowercase all letters
    line = line.lower()
    # remove replace digits with 0
    line = re.sub('[1-9]', '0', line)
    # remove non-English alphabet, spaces, or .
    return re.sub('[^a-z0.\s#]', '', line)


### Q2

'''
The estimation method uses a base of maximum likelihood estimation (MLE) by simply
counting the number of observations of a specific 3-char str and dividindg by the 
total number of 3-char strs in the corpus. However, there are no probabilities equal
to zero, so this means that some sort of smoothing was used. I do not belive this is
only add-alpha smoothing because we do not see the same probability for all unseen
events. The most common probability is 3.333e-02, but there are numerous other 
repeated probabilities as well. I believe this is backoff but need to do more analysisto show that it's not interpolation.  
'''


### Q3


def write_to_file(model, file_name):
    with open(file_name, 'w') as f:
        dict_sorted = dict(sorted(model.items()))
        for key, value in dict_sorted.items():
            f.write(key + '    ' + str(value))
            f.write('\n')


def generate_counts(infile, n):
    '''
    Returns of dictionary of counts for each unique n-long character sequence for a 
    give text file.

    Parameters:
        infile (text file): Text file to use as training data 
        n (int): desired size of n_gram

    Returns: 
        n_counts (dict): dictionary of counts for each character sequence
    '''
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
        all_combos = [a for a in all_combos]
    if n == 2:
        all_combos = [a + b for a, b in all_combos]
    if n == 3:
        all_combos = [a + b + c for a, b, c in all_combos]

    all_combo_dict = dict.fromkeys(all_combos, 0)
    all_combo_dict.update(n_counts)
    return all_combo_dict

def ngram_model(infile, n, add_alpha=None):
    '''
    Creates an n-gram character language model using a training data set and outputs 
    a dictionary of the model probabilities.   

    Parameters:
        infile (text file): Text file to use as training data 
        n (int): desired size of n_gram
        add_alpha (float): optional function parameter that implement add-alpha 
        smoothing. 
    
    Returns:
        norm_counts (dict): all n-character sequences with estimated normalized
        probabilities 
    '''
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
            # vocab_size = len(generate_counts(infile, 1))

    norm_counts = {}
    n_counts = generate_counts(infile, n)
    if n > 1:
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
            # unigram model doesn't need counts from n-1 gram model
    else:
        sum_counts = sum(n_counts.values())
        norm_counts = {key: (value + alpha) / (sum_counts + alpha * vocab_size)
                       for key, value in n_counts.items()}

    write_to_file(norm_counts, str(n) + '-gram.txt')
    return norm_counts


def perplexity(infile, n_gram_model, n):
    p_log = 0
    n_gram_count = 0
    for line in infile:
        line = preprocess_line(line)
        for j in range(len(line) - (n-1)):
            n_gram = line[j:j + n]
            n_gram_count += 1
            p_log += np.log2(n_gram_model[n_gram])
    return 2 ** (-1 / n_gram_count * p_log)


def generate_text(model, n, length):
    text = max(model, key=model.get)
    while len(text) < length:
        condition = text[-(n-1):]
        print('full text: ', text)
        print('condition: ', condition)
        sub_dict = {key: value for key, value in model.items() if key.startswith(condition)}
        print('max key: ', max(sub_dict, key=sub_dict.get))
        text += max(sub_dict, key=sub_dict.get)[-1]


data = open('training.en.txt').read().splitlines()
data_train, data_valid = train_test_split(data, test_size=0.2, shuffle=True)
data_valid, data_test = train_test_split(data_valid, test_size=0.5, shuffle=True)

trigram = ngram_model(data_train, 3, add_alpha=0.07)
# print(perplexity(data_train, trigram, 3))
# print(perplexity(data_valid, trigram, 3))
#sub_dict = {key: value for key, value in trigram.items() if key.startswith('ng') }
#write_to_file(sub_dict, 'ngdict.txt')

generate_text(trigram, 3, 20)


# interpolation test
# lambdas = [0.5, 0.25, 0.25]
# n = 3
# ngram_array = [dict() for i in range(n)]
# for i in range(n):
#     ngram_array[i] = ngram_model(data_train, i + 1)
#
# model = {}
# p_log = 0
# n_gram_count = 0
# for line in data_valid:
#     line = preprocess_line(line)
#     for j in range(len(line) - (n)):
#         n_gram = line[j:j + n]
#         n_gram_count += 1
#         probability = 0
#         for i in range(n):
#             if n_gram[:i + 1] in ngram_array[i]:
#                 probability += lambdas[i] * ngram_array[i][n_gram[:i + 1]]
#         model[n_gram] = probability
#         p_log += np.log2(probability)

# write_to_file(model, 'interpolation.txt')
# print(2**(-1/n_gram_count * p_log))
