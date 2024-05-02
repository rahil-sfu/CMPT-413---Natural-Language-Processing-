import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10



#### Word Segmentation (p. 223)

class Segment:
    
    def __init__(self, Pw):
        self.Pw = Pw
        
    def __eq__(self, other):
        return self.key == other.key
    
    def segmentIterative(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []

        chart = []
        heap_queue = []
        words = [ w for w in text ] 
        for word in words:
            if (word == text[0]):
                    #print("Word[i]=",word)
                    print(self.logPwords(word))
                    heap_queue.append([word, 0, self.logPwords(word), None])
                    heapq.heapify(heap_queue)
        i = 1   
        for word in text:
            chart.append([word, self.logPwords(word)])
            i+=1
        
        #heap = [(item[2], item) for item in heap_queue]
        # # min heap ordered (lPW, chartEntry)

        while(len(heap_queue) > 0):
            entry = heapq.heappop(heap_queue)
            endindex = len(chart) - 1
            #print(endindex)
            if (len(chart) != 1):
                preventry = chart[endindex-1]
                if(entry[2] > preventry[1]):
                    chart[endindex] = entry

                elif(entry[2] <= preventry[1]):
                    continue
                              
            else:
                chart[endindex] = entry
                
            for newword in text:
                if (newword == text[endindex]):
                    newentry = [newword, endindex+1, entry[2] + self.logPwords(newword), entry]
                    if(not (newentry in heap_queue)):
                        heapq.heappush(heap_queue, newentry)
        
        #finalindex = len(text)    
        #finalentry = chart[finalindex]
        size = len(chart)
        for i in range(size):
            str = " ".join(chart[i][0])
        return str

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

    def logPwords(self, words): 
        # "The Naive Bayes probability of a sequence of words."
        return sum(self.Pw(w) for w in words)
#### Support functions (p. 224)

#def add(nums):
    # "Return the product of a sequence of numbers."
    #return reduce(operator.add, nums, 1)

def product(nums):
    # "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    # "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key):   
        if key in self: return log10(self[key])/self.N  # added log here
        else: return self.missingfn(key, self.N)

# Parses the input datafile.
def datafile(name, sep='\t'):
    # "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)
#bigram           
def cPw(word, prev):
    try:
        return P2w[prev + ' ' + word]/float(Pw[prev])
    except KeyError:
        return Pw(word)
    
def avoid_long_words(word, N):
    return 10./(N * 10**len(word))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)
    
    #Pw = Pdist(data=datafile(opts.counts1w),N=1024908267229, missingfn = avoid_long_words)
    Pw = Pdist(data=datafile(opts.counts1w))
    segmenter = Segment(Pw)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segmentIterative(line.strip())))
