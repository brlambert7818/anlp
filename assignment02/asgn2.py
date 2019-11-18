from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
import json
import collections

from load_map import *

STEMMER = PorterStemmer()


def w_count(word):
    # helper function to get the count of a word (string)
    return o_counts[word2wid[word]]


def tw_stemmer(word):
    '''Stems the word using Porter stemmer, unless it is a
    username (starts with @).  If so, returns the word unchanged.

    :type word: str
    :param word: the word to be stemmed
    :rtype: str
    :return: the stemmed word

    '''

    if word[0] == '@': #don't stem these
        return word
    else:
        return STEMMER.stem(word)


def pmi(c_xy, c_x, c_y, N = None):
    '''Compute the  pointwise mutual information using cooccurrence counts.

    :type c_xy: int
    :type c_x: int
    :type c_y: int
    :type N: int
    :param c_xy: coocurrence count of x and y
    :param c_x: occurrence count of x
    :param c_y: occurrence count of y
    :param N: total observation count
    :rtype: float
    :return: the pmi value
    '''
    return np.log2(N*c_xy / (c_x*c_y))


def pmi_smooth(c_xy, c_x, c_y, N):
    '''Compute the  pointwise mutual information using cooccurrence counts.

    :type c_xy: int
    :type c_x: int
    :type c_y: int
    :type N: int
    :param c_xy: coocurrence count of x and y
    :param c_x: occurrence count of x
    :param c_y: occurrence count of y
    :param N: total observation count
    :rtype: float
    :return: the pmi value
    '''
    return np.log2((c_xy/N) / ((c_x/N)*c_y))


#Do a simple error check using value computed by hand
if pmi(2,4,3,12) != 1: # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")


def cos_sim(v0, v1):
    '''Compute the cosine similarity between two sparse vectors.

    :type v0: dict
    :type v1: dict
    :param v0: first sparse vector
    :param v1: second sparse vector
    :rtype: float
    :return: cosine between v0 and v1
    '''
    # We recommend that you store the sparse vectors as dictionaries
    # with keys giving the indices of the non-zero entries, and values
    # giving the values at those dimensions.
    dot_prod = 0
    for k, v in v0.items():
        if k in v1.keys():
            dot_prod += v*v1[k]
    norms = np.linalg.norm(list(v0.values())) * np.linalg.norm(list(v1.values()))
    return 0 if norms == 0 else dot_prod / norms


def create_ppmi_vectors(wids, o_counts, co_counts, tot_count, alpha = None):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for wid0 in wids:
        # count of target word
        c_wid0 = o_counts[wid0]
        wid1_dict = {}
        for wid1 in co_counts[wid0]:
            if wid0 != wid1:
                # count of context word
                c_wid1 = o_counts[wid1]
                # co-occurence counts of target and context word
                co_count = co_counts[wid0][wid1]
                pmi_temp = pmi(co_count, c_wid0, c_wid1, tot_count)
                # positive PMI with sparse vector representation
                if pmi_temp > 0:
                    wid1_dict[wid1] = pmi_temp
        vectors[wid0] = wid1_dict
    return vectors


def create_ppmi_vectors_smooth(wids, o_counts, co_counts, tot_count, alpha):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for wid0 in wids:
        # count of target word
        c_wid0 = o_counts[wid0]

        wid1_dict = {}
        c_wid1s = []
        wid1s = co_counts[wid0].keys()
        for w in wid1s:
            c_wid1s.append(o_counts[w])
        # sum_smooth = np.sum(np.array(list(o_counts.values())) ** alpha)
        sum_smooth = np.sum(np.array(c_wid1s)** alpha)
        for wid1 in co_counts[wid0]:
            if wid0 != wid1:
                # count of context word
                p_wid1 = o_counts[wid1]**alpha / sum_smooth
                # co-occurence counts of target and context word
                co_count = co_counts[wid0][wid1]
                pmi_temp = pmi_smooth(co_count, c_wid0, p_wid1, tot_count)
                # positive PMI with sparse vector representation
                if pmi_temp > 0:
                    wid1_dict[wid1] = pmi_temp
        vectors[wid0] = wid1_dict
    return vectors

def log2fac(k):
    return log((2*np.pi*k)**(.5),2) + k*log(k/np.exp(1), 2)

def p_binom(p, k, n):
    #return k*log(x) + (n-k)*log(1-x) + log2fac(n) - log2fac(n-k) - log2fac(k)
    return k*log(p) + (n-k)*log(1-p) 

def ted_dunning(wids, o_counts, co_counts, N):
    vectors = {}
    for wid0 in wids:
        # count of target word
        c_wid0 = o_counts[wid0]

        c_wid1s = []
        wid1s = co_counts[wid0].keys()
        for w in wid1s:
            c_wid1s.append(o_counts[w])
        wid1_dict = {}
        for wid1 in [8, 999, 140, 77, 926]:
            if wid0 != wid1:
                # N = np.sum(c_wid1s)
                # count of context word
                c_wid1 = o_counts[wid1]
                # co-occurence counts of target and context word
                co_count = co_counts[wid0][wid1]
                p = co_count / N      
                p1 = c_wid0 / N
                p2 = c_wid1 / N

                b1 = p_binom(p, c_wid0, N)
                b2 = p_binom(p, c_wid1, N)
                b3 = p_binom(p1, c_wid0, N)
                b4 = p_binom(p2, c_wid1, N)

                wid_ll = (b3 + b4 - b1 - b2)
                wid1_dict[wid1] = wid_ll
        vectors[wid0] = wid1_dict
    return vectors


def read_counts(filename, wids):
    '''Reads the counts from file. It returns counts for all words, but to
    save memory it only returns cooccurrence counts for the words
    whose ids are listed in wids.

    :type filename: string
    :type wids: list
    :param filename: where to read info from
    :param wids: a list of word ids
    :returns: occurence counts, cooccurence counts, and tot number of observations
    '''
    o_counts = {} # Occurence counts
    co_counts = {} # Cooccurence counts
    fp = open(filename)
    N = float(next(fp))
    for line in fp:
        line = line.strip().split("\t")
        wid0 = int(line[0])
        o_counts[wid0] = int(line[1])
        if wid0 in wids:
            co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
    return o_counts, co_counts, N


def print_sorted_pairs(similarities, o_counts, first=0, last=100):
    '''Sorts the pairs of words by their similarity scores and prints
    out the sorted list from index first to last, along with the
    counts of each word in each pair.

    :type similarities: dict
    :type o_counts: dict
    :type first: int
    :type last: int
    :param similarities: the word id pairs (keys) with similarity scores (values)
    :param o_counts: the counts of each word id
    :param first: index to start printing from
    :param last: index to stop printing
    :return: none
    '''
    if first < 0: last = len(similarities)
    for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
        word_pair = (wid2word[pair[0]], wid2word[pair[1]])
        print("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair], str(word_pair),
                                         o_counts[pair[0]], o_counts[pair[1]]))

def freq_v_sim(sims):
    xs = []
    ys = []
    for pair in sims.items():
        ys.append(pair[1])
        c0 = o_counts[pair[0][0]]
        c1 = o_counts[pair[0][1]]
        xs.append(min(c0,c1))
    plt.clf() # clear previous plots (if any)
    plt.xscale('log') #set x axis to log scale. Must do *before* creating plot
    plt.plot(xs, ys, 'k.') # create the scatter plot
    plt.xlabel('Min Freq')
    plt.ylabel('Similarity')
    print("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
    # plt.show() #display the set of plots

def make_pairs(items):
    '''Takes a list of items and creates a list of the unique pairs
    with each pair sorted, so that if (a, b) is a pair, (b, a) is not
    also included. Self-pairs (a, a) are also not included.

    :type items: list
    :param items: the list to pair up
    :return: list of pairs

    '''
    return [(x, y) for x in items for y in items if x < y]


#test_words = ["cat", "dog", "mouse", "computer", "@justinbieber"]
## test_words = ["cat", "dog"]
#stemmed_words = [tw_stemmer(w) for w in test_words]
#all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them
#
## you could choose to just select some pairs and add them by hand instead
## but here we automatically create all pairs 
#wid_pairs = make_pairs(all_wids)

# (o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/lab8/counts", all_wids)
#(o_counts, co_counts, N) = read_counts("/Users/brianlambert/tweets_2011/counts", all_wids)

# PMI
# vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)
# c_sims = {(wid0,wid1):  cos_sim(vectors[wid0], vectors[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by cosine similarity")
# print_sorted_pairs(c_sims, o_counts)
# print("=====================================================")

# PMI smoothed
# vectors = create_ppmi_vectors_smooth(all_wids, o_counts, co_counts, N, 2)
# c_sims = {(wid0,wid1): cos_sim(vectors[wid0], vectors[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by cosine similarity")
# print_sorted_pairs(c_sims, o_counts)

# Dunning G2
#vectors = ted_dunning(all_wids, o_counts, co_counts, N)
#c_sims = {(wid0,wid1): vectors[wid0][wid1] for (wid0,wid1) in wid_pairs}
##c_sims = {(wid0,wid1): cos_sim(vectors[wid0], vectors[wid1]) for (wid0,wid1) in wid_pairs}
#print("Sort by cosine similarity")
#print_sorted_pairs(c_sims, o_counts)

###################### DATA AGGREGATION #########################


def get_counts(count):
    all_wids = wid2word.keys()
    (o_counts, co_counts, N) = read_counts("/Users/brianlambert/tweets_2011/counts", all_wids)

    co_counts_filter = {}
    for k1, v1 in co_counts.items():
        d_inner = {}
        for k2, v2 in v1.items():
            if v2 == count:
                d_inner[k2] = v2
        if len(d_inner.keys()) > 0:
            co_counts_filter[k1] = d_inner

    o_counts_filter = {}
    for k in co_counts_filter.keys():
        o_counts_filter[k] = o_counts[k]
    return o_counts_filter, co_counts_filter



def get_sorted_c(unsorted_dict):

    sorted_x = sorted(unsorted_dict.items(), key=lambda kv: kv[1])
    sorted_dict = collections.OrderedDict(sorted_x)
    # for k,v in sorted_dict.items():
    #     print(wid2word[k] + ': ' + str(v))
    return sorted_dict


def get_co_range(lower, upper, counts_filter, co_counts_filter):
    
    temp_counts = {}
    for k,v in counts_filter.items():
        if lower <= counts_filter[k] <= upper:
            temp_counts[k] = v

    co_occurs = {}
    for k1, v1 in temp_counts.items():
        w1s = []
        for k2, v2 in temp_counts.items():
            if k1 != k2:
                if k2 in co_counts_filter[k1].keys():
                    w1s.append(k2)
        if len(w1s) > 0:
            co_occurs[k1] = w1s
    return co_occurs

# word picking helper code
o_counts_filter, co_counts_filter = get_counts(200)
#x1 = get_co_range(200, 2000, counts_filter, co_counts_filter)
#x2 = get_co_range(20000, 40000, counts_filter, co_counts_filter)
#x3 = get_co_range(200000, 250000, counts_filter, co_counts_filter)
#x1_sort = get_sorted_c(x1)
#x2_sort = get_sorted_c(x2)
#x3_sort = get_sorted_c(x3)
#
#
#for k,v in x3_sort.items():
#    print(wid2word[k] + ': ' + str(o_counts[k]))
#    print('')
#    for i in v:
#        print(wid2word[i] + ': ' + str(o_counts[i]))
#    print('----------------')
    

def get_similarity(sim_fn, o_counts, co_counts, N, all_wids, wid_pairs):
    sims = np.zeros((len(wid_pairs), 6))

    
    i = 0
    for pair in wid_pairs:
        wid0 = pair[0]
        wid1 = pair[1]
        print(wid0)
        print(wid1)
        print('---------------')

        vectors = None
        if sim_fn == 'g2':
            vectors = ted_dunning(all_wids, o_counts, co_counts, N)
            
        elif sim_fn == 'pmi':
            vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)
        elif sim_fn == 'pmi_smooth':
            vectors = create_ppmi_vectors_smooth(all_wids, o_counts, co_counts, N, 0.75)
        else:
            raise Exception('Invalid similarity measure')
            
        sims[i, 0] = vectors[wid0][wid1]
        sims[i, 1] = o_counts[wid0]
        sims[i, 2] = o_counts[wid1]
        sims[i, 3] = co_counts[wid0][wid1]
        sims[i, 4] = wid0
        sims[i, 5] = wid1
        
        i += 1
    return sims

N = 138489679
all_test_wids1 = {word2wid['dystopia'], word2wid['wubfur'], 
                 word2wid['faidzin'], word2wid['aidin'], 
                 word2wid['@2ksports'], word2wid['#nba2k12'], 
                 word2wid['@appwill'], word2wid['retina'], 
                 word2wid['#phi'], word2wid['#philadelphia']}
                          
test_pairs1 =    [(word2wid['dystopia'], word2wid['wubfur']), 
                 (word2wid['faidzin'], word2wid['aidin']), 
                 (word2wid['@2ksports'], word2wid['#nba2k12']), 
                 (word2wid['@appwill'], word2wid['retina']), 
                 (word2wid['#phi'], word2wid['#philadelphia'])]
                           
sim_pmi = get_similarity('pmi', o_counts_filter, co_counts_filter, N, all_test_wids1, test_pairs1)

vectors = create_ppmi_vectors(all_test_wids1, o_counts_filter, co_counts_filter, N)
sim_pmis = get_similarity('pmi_smooth', o_counts_filter, co_counts_filter, N, all_test_wids1, test_pairs1)

sim_g2 = get_similarity('g2', o_counts_filter, co_counts_filter, N, all_test_wids1, test_pairs1)

sim1 = np.append([sim_g2[:, 0], sim_pmis[:, 0], sim_pmi], axis=1)


###############################################################################


all_test_wids2 = {word2wid['rat'], word2wid['mous'], 
                 word2wid['grin'], word2wid['giggl'], 
                 word2wid['lisa'], word2wid['simpson'], 
                 word2wid['belt'], word2wid['bibl'], 
                 word2wid['virtual'], word2wid['microsoft']}

test_pairs2 =    [(word2wid['rat'], word2wid['mous']), 
                 (word2wid['grin'], word2wid['giggl']), 
                 (word2wid['lisa'], word2wid['simpson']), 
                 (word2wid['belt'], word2wid['bibl']), 
                 (word2wid['virtual'], word2wid['microsoft'])]
                           
sim = get_similarity('pmi', o_counts_filter, co_counts_filter, N, all_test_wids2, test_pairs2)
      

            
        

