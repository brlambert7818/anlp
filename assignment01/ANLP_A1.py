#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:27:22 2019

@author: onyskj
"""

import re
import os
import numpy as np
from sklearn.model_selection import train_test_split
import itertools


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
    line = re.sub('[1-9]', '0',line) 
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
repeated probabilities as well. I believe this is backoff but need to do more analysis 
to show that it's not interpolation.  
'''


### Q3


def write_to_file(model, file_name):
    '''
    '''
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
    #with open(infile, 'r') as f:
        #for line in f:
    for line in infile:
        line = preprocess_line(line) 
        for j in range(len(line)-(n)):   
            n_gram = line[j:j+n]
            if n_gram in n_counts:
                n_counts[n_gram] += 1
            else:
                n_counts[n_gram] = 1
    return n_counts


def ngram_model(infile, n, add_alpha=None, fillIn=False):
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
    if add_alpha != None:
        if add_alpha > 1:
            raise Exception('Alpha must be <= 1')
        else:
            alpha = add_alpha 
            vocab_size = 29# len(generate_counts(infile, 1))

    norm_prob = {} 
    n_counts = generate_counts(infile, n) 
    if n > 1:
       n_minus_counts = generate_counts(infile, n-1)
       # find the counts for the n-1 gram model to use for the divisor in 
       # probability calculation
       for key, value in n_counts.items():
           n_minus_key = key[:n-1]
           norm_prob[key] = (n_counts[key] + alpha) \
                              / (n_minus_counts[n_minus_key] + alpha*vocab_size) 
    # unigram model doesn't need counts from n-1 gram model
    else:
        sum_counts = sum(n_counts.values())
        norm_prob = {key: (value + alpha)  / (sum_counts + alpha*vocab_size) \
                             for key, value in n_counts.items()}  

    # fill out missing probabilities if alpha smoothing
    if n ==3 and alpha!=0 and fillIn:
        all_trigrams = set([''.join(x) for x in itertools.product(' 0.abcdefghijklmnopqrstuvwxyz', repeat=3)])
        missing_trigrams = all_trigrams - set(norm_prob)
        
        for element in missing_trigrams:
            n_minus_element = element[:n-1]
            if n_minus_element not in n_minus_counts:
                norm_prob[element] = (alpha)/(alpha*vocab_size)
            else:
                norm_prob[element] = (alpha)/(n_minus_counts[n_minus_element]+alpha*vocab_size)
    
    #write_to_file(norm_counts, str(n) + '-gram.txt')
    if add_alpha != None:
        return norm_prob#, n_minus_counts 
    else:
        return norm_prob

def perplexity(infile, n_gram_model, n, nminus_counts=None, add_alpha=None):
    '''
    '''
    p_log = 0
    n_gram_count = 0
    for line in infile:
        line = preprocess_line(line) 
        for j in range(len(line)-(n)):
            n_gram = line[j:j+n]
            n_gram_count += 1
            if n_gram in n_gram_model:
                p_log += np.log2(n_gram_model[n_gram])
            elif add_alpha != None:
                n_minus_key = n_gram[:n-1]
                nminus_count = 0 
                if n_minus_key in nminus_counts.keys():
                    nminus_count = nminus_counts[n_minus_key]
                p_log += np.log2(add_alpha / (nminus_count + 
                                         (add_alpha*n_gram_count)) )
            else:
                raise Exception('Zero probability value. Implement smoothing.')
    return 2**(-1/n_gram_count * p_log) 

def smoothed_perplex(infile, n_gram_model):
    p_log = 0
    n_gram_count = 0
    for line in infile:
        line = preprocess_line(line) 
        for j in range(len(line)-(n)):
            n_gram = line[j:j+n]
            n_gram_count += 1
            if n_gram in n_gram_model:
                p_log += np.log2(n_gram_model[n_gram])
                
    return 2**(-1/n_gram_count * p_log)

def interpolation(infile, n, lambdas):
    '''
    Probability smoothing technique that uses weighted sums of all n, n-1, ..., 1
    gram language models
    Parameters
        infile (text file): Text file to use as training data 
        n (int): desired size of n_gram
        lambdas (array): weights to apply to each n-gram probability
    Returns:
        inter_norm (dict): all n-character sequences with estimated normalized
        probabilities using interpolation smoothing 
    '''
    inter_norm = {}
    # construct array of all n, n-1, ..., 1 gram models
    ngram_array = [dict() for i in range(n)] 
    for i in range(n):
        ngram_array[i] = ngram_model(infile, i+1)

    for key, value in ngram_array[n-1].items():
        # add first weighted probability (n-gram model)
        norm_value = lambdas[n-1]*value

        # find n-1, n-2, ..., 1 gram models and sum weighted probabilities 
        for i in range(n-1):
            # substring of character sequence
            n_minus_key = key[:i+1]
            n_minus_probability = ngram_array[i][n_minus_key]
            norm_value += lambdas[i]*n_minus_probability
        # add full weighted sum for a single n-character sequence to final dict
        inter_norm[key] = norm_value

        #write_to_file(inter_norm)
        return inter_norm


data = open('training.en').read().splitlines()
data_train, data_valid = train_test_split(data, test_size=0.2, shuffle=True)
data_valid, data_test = train_test_split(data_valid, test_size=0.5, shuffle=True)


lambdas = [0.5, 0.25 ,0.25]
n = 3
ngram_array = [dict() for i in range(n)] 
for i in range(n):
    ngram_array[i] = ngram_model(data_train, i+1)

model = {}
p_log = 0
n_gram_count = 0
for line in data_valid:
    line = preprocess_line(line) 
    for j in range(len(line)-(n)):
        n_gram = line[j:j+n]
        n_gram_count += 1
        probability = 0
        for i in range(n):
            if n_gram[:i+1] in ngram_array[i]:
                probability += lambdas[i]*ngram_array[i][n_gram[:i+1]]
        model[n_gram] = probability 
        p_log += np.log2(probability)

write_to_file(model, 'interpolation.txt')
print(2**(-1/n_gram_count * p_log)) 


'''
trigram, bigram_counts = ngram_model(data_train, 3, add_alpha=1)
print(perplexity(data_train, trigram, 3, bigram_counts, add_alpha=1))
print(perplexity(data_valid, trigram, 3, bigram_counts, add_alpha=1))
'''

trigram = ngram_model(data_train, 3)

print(perplexity(data_train, trigram, 3))
print(perplexity(data_valid, trigram, 3))


# Check for intersection of train and valid sets model
train_modelSet = set(ngram_model(data_train,3))
valid_modelSet = set(ngram_model(data_valid,3))
#for name in train_modelSet.intersection(valid_modelSet):
#    print (name, ngram_model(data_train,3)[name])


#Train on different alphas and compute perplexity on dev set for 
alphas = np.linspace(0.00001,1,100)
alpha_smoothed_models=[dict() for i in range(len(alphas))] 
perplexities=np.zeros(len(alphas))
for i in range(len(alphas)):
    alpha_smoothed_models[i] = ngram_model(data_train,3,alphas[i],True)
    perplexities[i] = smoothed_perplex(data_valid, alpha_smoothed_models[i]) #under validation

alphas[perplexities.argmin()]

#
#alpha=0.9
#myTriFilled = ngram_model(data_train,3,alpha,True)
#myTri = ngram_model(data_train,3,alpha)
#
##smoothed_perplex(data_train, myTriFilled)
#smoothed_perplex(data_valid, myTriFilled)

