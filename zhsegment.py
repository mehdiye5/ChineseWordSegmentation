import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10

class Segment:
    def __init__(self, Pw):
        self.Pw = Pw

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: 
            return []
        segmentation = [ w for w in text ] # segment each char into a word
        return segmentation

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

class UnigramSegmenter:
    def __init__(self, Pw):
        self.Pw = Pw
        self.maxlen = 7 # Max word length

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: 
            return []
        return self.dynamic_segment(text)

    def dynamic_segment(self, text):
        chart = {}
        wordheap = []

        # Initialize heap with all possible starting words
        for i in range(min(len(text), self.maxlen)):
            word = text[:i+1]
            prob = math.log10(self.Pw(word))

            # Probability as first element for key in _heapify_max() function
            entry = (prob, word, len(word), None)
            heapq.heappush(wordheap, entry)

        # Iterate until empty heap
        while wordheap:

            # Maxheap and pop         
            heapq._heapify_max(wordheap)
            entry = heapq.heappop(wordheap)
            prob, word, wordlen, _ = entry
            endidx = wordlen - 1 # Get last char pos in word
            
            # Checking if the current word prob is greater than past w/ endindex
            # If yes, update chart
            if endidx in chart:
                if prob > chart[endidx][0]:
                    chart[endidx] = entry
                else:
                    continue
            else:
                chart[endidx] = entry

            # Get all possbile words that start after current segment
            for i in range(endidx + 1, endidx + 1 + min(len(text[endidx+1:]), self.maxlen)):
                
                # Create new entry
                new_word = text[endidx+1:i+1]
                new_prob = math.log10(self.Pw(new_word))
                new_entry = (prob + new_prob, new_word, wordlen+len(new_word), endidx)

                # Add entry if not in heap
                if new_entry not in wordheap:
                    heapq.heappush(wordheap, new_entry)

        # Get line segment
        segments = []
        bp = len(text) - 1 
        while bp is not None:
            segments.append(chart[bp][1])
            bp = chart[bp][3]

        return segments[::-1]

class BigramSegmenter:
    def __init__(self, pwunigram, pwbigram):
        self.pwunigram = pwunigram
        self.pwbigram = pwbigram
        self.maxlen = 7 # Max word length
        self.alpha = 1.02 # Scale factor when word combination not in bigram counts

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: 
            return []
        return self.dynamic_segment(text)

    def bigram_backoff(self, prev_word, new_word):
        # Backoff formula: http://www.cs.cornell.edu/courses/cs4740/2014sp/lectures/smoothing+backoff.pdf
        # Combining words for bigram lookup
        bigram = prev_word + " " + new_word

        # If combination in bigrams else use unigram
        if bigram in self.pwbigram and prev_word in self.pwunigram:
            return math.log10(self.pwbigram(bigram) / self.pwunigram(prev_word))

        # Scale by alpha when bigram not found
        elif new_word in self.pwunigram:
            return 1.02 * math.log10(self.pwunigram(new_word))

        return math.log10(self.pwunigram(new_word))

    def dynamic_segment(self, text):
        chart = {}
        wordheap = []

        # Initialize heap with all possible starting words
        for i in range(min(len(text), self.maxlen)):
            word = text[:i+1]
            prob = self.bigram_backoff("<S>", word) # padding for startword

            # Probability as first element for key in _heapify_max() function
            entry = (prob, word, len(word), None)
            heapq.heappush(wordheap, entry)

        # Iterate until empty heap
        while wordheap:

            # Heapify and pop         
            heapq._heapify_max(wordheap)
            entry = heapq.heappop(wordheap)
            prob, word, wordlen, _ = entry
            endidx = wordlen - 1 # Get last char pos in word
            
            # Checking if the current word prob is greater than past w/ endindex
            # If yes, update chart
            if endidx in chart:
                if prob > chart[endidx][0]:
                    chart[endidx] = entry
                else:
                    continue
            else:
                chart[endidx] = entry

            # Get all possbile words that start after current segment
            for i in range(endidx + 1, endidx + 1 + min(len(text[endidx+1:]), self.maxlen)):
                
                # Create new entry
                new_word = text[endidx+1:i+1]

                # Getting previous word using endidx
                if chart[endidx][3] is not None:
                    prev_word = chart[endidx][1]
                else:
                    prev_word = "<S>"

                # Getting prob w/ backoff
                new_prob = self.bigram_backoff(prev_word, new_word)
                new_entry = (entry[0] + new_prob, new_word, entry[2] + len(new_word), endidx)

                # Add entry if not in heap
                if new_entry not in wordheap:
                    heapq.heappush(wordheap, new_entry)

        # Get line segment
        segments = []
        bp = len(text) - 1 
        while bp is not None:
            segments.append(chart[bp][1])
            bp = chart[bp][3]

        return segments[::-1]

#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

def avoid_long_words(word, N):
    "Estimate the probability of an unknown word."
    return 10./(N*10000**len(word))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    pwunigram = Pdist(data=datafile(opts.counts1w), missingfn=avoid_long_words)
    pwbigram = Pdist(data=datafile(opts.counts2w))

    u_segmenter = UnigramSegmenter(pwunigram)
    b_segmenter = BigramSegmenter(pwunigram, pwbigram)


    with open(opts.input) as f:
        for line in f:
            # arr = " ".join(u_segmenter.segment(line.strip()))
            arr = " ".join(b_segmenter.segment(line.strip()))
            print(arr)
