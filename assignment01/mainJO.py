'''
@ author: Brian Lambert, Jakub Onysk
'''

import re
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
    if (not line.startswith('#')) and (not line.endswith('#')):
        line = '#'+line+'#'
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

    all_combos = product('#abcdefghijklmnopqrstuvwxyz0. ', repeat=n)
    if n == 1:
        all_combos = [a for a in all_combos]
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
            vocab_size = 30

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

line = '##abaab#'

def find_optimal_alpha(alphas, validation_set, train_set, n):
    '''
    Trains a model on train data for different alphas and then calculates 
    a perplexity on a validation set under this model to then pick alpha 
    that gives the smallest perplexity.  
    '''
    if any(alphas<=0) or any(alphas>1):
        raise Exception('Alpha should be > 0 and <=1')
    alpha_smoothed_models=[dict() for i in range(len(alphas))] 
    perplexities=np.zeros(len(alphas))
    for i in range(len(alphas)):
        alpha_smoothed_models[i] = ngram_model(train_set,n,alphas[i])
        perplexities[i] = perplexity(validation_set, alpha_smoothed_models[i],n) #under validation

    return alphas[perplexities.argmin()]

## English data ##
data = open('training.en').read().splitlines()
data_train, data_valid = train_test_split(data, test_size=0.2, shuffle=True)
data_valid, data_test = train_test_split(data_valid, test_size=0.5, shuffle=True)

perplexity(data_test,ngram_model(data_train,3, add_alpha=1),3)

# Find best alpha for English #
alphas = np.linspace(0.0000001,1,1000)
#best_alpha = np.rouwnd(find_optimal_alpha(alphas, data_valid, data_train, 3),4)
best_alpha = 0.047

# Trigram model + save to file #
trigram = ngram_model(data_train, 3, add_alpha=best_alpha)
write_to_file(trigram, 'trigram.txt')

# Find two-character history 'ng' #
ng_ngrams = {key: value for key, value in trigram.items() if key.startswith('ng') }
write_to_file(ng_ngrams, 'ngdict.txt')

#perplexity(data_valid, trigram, 3)
#print(perplexity(data_train, trigram, 3))
print(perplexity(data_test, trigram, 3))


## German data ##
dataDe = open('training.de').read().splitlines()
data_trainDe, data_validDe = train_test_split(dataDe, test_size=0.2, shuffle=True)
data_validDe, data_testDe = train_test_split(data_validDe, test_size=0.5, shuffle=True)

# Find best alpha for German #
alphas = np.linspace(0.0000001,1,1000)
#best_alphaDe = np.round(find_optimal_alpha(alphas, data_validDe, data_trainDe, 3),4)
best_alphaDe = 0.1041

# Trigram model
trigramDe = ngram_model(data_trainDe, 3, add_alpha=best_alphaDe)

#print(perplexity(data_trainDe, trigramDe, 3))
#print(perplexity(data_validDe, trigramDe, 3))
#print(perplexity(data_testDe, trigramDe, 3))

## Spanish data ##
dataEs = open('training.es').read().splitlines()
data_trainEs, data_validEs = train_test_split(dataEs, test_size=0.2, shuffle=True)
data_validEs, data_testEs = train_test_split(data_validEs, test_size=0.5, shuffle=True)

# Find best alpha for Spanish #
alphas = np.linspace(0.0000001,1,1000)
#best_alphaEs = np.round(find_optimal_alpha(alphas, data_validEs, data_trainEs, 3),4)
best_alphaEs = 0.0781


# Trigram model #
trigramEs = ngram_model(data_trainEs, 3, add_alpha=best_alphaEs)

#print(perplexity(data_trainEs, trigramEs, 3))
#print(perplexity(data_validEs, trigramEs, 3))
#print(perplexity(data_testEs, trigramEs, 3))


### Q4 ###

def generate_text(model, n, length, br):
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
        if not br:
            if bins[-1] != 1:
                bins[-1] = 1
        sample = outcomes[np.digitize(np.random.random_sample(), bins)][-1]
        text += sample

    print('Generated Text: ', text)
#    return text

def input_model(infile):
    model = {}
    data = open(infile).read().splitlines()
    for line in data:
        (key, value) = line.split('\t')
        model[key] = float(value)
    return model

## Generate from trigram model ##
generate_text(trigram, 3, 300, br=False)

## Generate from model-br ##
model_br = input_model('model-br.en')
generate_text(model_br, 3, 300, br=True)
print(perplexity(data_valid, model_br, 3))

## Other langs ## 
generate_text(trigramEs,3,300,br=False)
generate_text(trigramDe,3,300,br=False)


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
                  
                  
### OTHER ###





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