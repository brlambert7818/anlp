'''
@ author: Brian Lambert
'''

import re
import os
import numpy as np


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
    return re.sub('[^a-z0.\s]', '', line)

    
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
    # file was reading in as TestIOWrapper instead of txt file
    # remove line 63 if file is successfully read in as raw txt
    cleanpath = os.path.abspath(infile)
    with open(cleanpath, 'r') as f:
        for line in f:
            line = preprocess_line(line) 
            for j in range(len(line)-(n)):
                n_gram = line[j:j+n]
                if n_gram in n_counts:
                    n_counts[n_gram] += 1
                else:
                    n_counts[n_gram] = 1
    return n_counts


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
    if add_alpha != None:
        if add_alpha > 1:
            raise Exception('Alpha must be <= 1')
        else:
            alpha = add_alpha 
            vocab_size = len(generate_counts(infile, 1))

    norm_counts = {} 
    n_counts = generate_counts(infile, n) 
    if n > 1:
       n_minus_counts = generate_counts(infile, n-1)
       # find the counts for the n-1 gram model to use for the divisor in 
       # probability calculation
       for key, value in n_counts.items():
           n_minus_key = key[:n-1]
           norm_counts[key] = (n_counts[key] + alpha) \
                              / (n_minus_counts[n_minus_key] + alpha*vocab_size) 
    # unigram model doesn't need counts from n-1 gram model
    else:
        sum_counts = sum(n_counts.values())
        norm_counts = {key: (value + alpha)  / (sum_counts + alpha*vocab_size) \
                             for key, value in n_counts.items()}  

    write_to_file(norm_counts, str(n) + '-gram.txt')
    return norm_counts 


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

        write_to_file(inter_norm)
            

def write_to_file(model, file_name):
    '''
    '''
    with open(file_name, 'w') as f:
        dict_sorted = dict(sorted(model.items()))
        for key, value in dict_sorted.items():
            f.write(key + '    ' + str(value))
            f.write('\n')
                


filepath = '/Users/brianlambert/Downloads/assignment1-data/training.en.txt'
#interpolation(filepath, 3, [0.5, 0.25, 0.25])
ngram_model(filepath, 3)

                




