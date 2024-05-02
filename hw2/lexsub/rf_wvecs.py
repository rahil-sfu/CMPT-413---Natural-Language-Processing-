import os, sys, optparse
import re
import math
import pymagnitude
import gzip
import math
import numpy
import re
import sys

from gensim.models import Word2Vec
from copy import deepcopy

isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()

''' Read all the word vectors and normalize them '''
def read_word_vecs(wvec_file):
  
  wordVecs = {}
  wordVectors = pymagnitude.Magnitude(wvec_file)
  
  for line in wordVectors:
    word = line[0]
    vecVal = line[1]
    wordVecs[word]= numpy.zeros(len(vecVal), dtype=float)
    for key, value in enumerate(line[1]):
      wordVecs[word][key] = float(value)    
    #wordVecs[word] /= math.sqrt((wordVecs[word]**2).sum() + 1e-6)
  sys.stderr.write("Vectors read from: "+wvec_file+" \n")
  
  return wordVecs

''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
  sys.stderr.write('\nWriting down the vectors in '+outFileName+'...')
  outFile = open(outFileName, 'w')  
  for word, values in wordVectors.items():
    #print(word, " ")
    outFile.write(word+' ')
    for val in wordVectors[word]:
      outFile.write('%.4f' %(val)+' ')
    outFile.write('\n') 
  sys.stderr.write('Complete \n')     
  outFile.close()
  
''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename):
  lexicon = {}
  for line in open(filename, 'r'):
    words = line.lower().strip().split()
    lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    
  sys.stderr.write("Word relations read from lexicon file: "+filename+" \n")
  return lexicon

''' Retrofit word vectors to a lexicon '''
def retrofit(wordVecs, lexicon, numIters):
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = wvVocab.intersection(set(lexicon.keys()))
  for it in range(numIters):
    # loop through every node also in ontology (else just use data estimate)
    for word in loopVocab:
      wordNeighbours = set(lexicon[word]).intersection(wvVocab)
      numNeighbours = len(wordNeighbours)
      #no neighbours, pass - use data estimate
      if numNeighbours == 0:
        continue
      # the weight of the data estimate if the number of neighbours
      newVec = numNeighbours * wordVecs[word]
      # loop over neighbours and add to new vector (currently with weight 1)
      for ppWord in wordNeighbours:
        newVec += newWordVecs[ppWord]
      newWordVecs[word] = newVec/(2*numNeighbours)
  
  sys.stderr.write("Retrofitting word vectors with lexicon completed for numIters = "+str(numIters)+" \n")
  return newWordVecs
  
if __name__=='__main__':
    
  optparser = optparse.OptionParser()
  optparser.add_option("-i", "--wordvecfile", dest="input", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
  optparser.add_option("-l", "--lexicon", dest="lexicon", default=os.path.join('data', 'lexicons', 'wordnet-synonyms+.txt'), help="word relation lexicon file")
  optparser.add_option("-o", "--output", dest="output", default=os.path.join('data', 'glove.6B.100d.retrofit.txt'), help="Output retrofitted word vecs text file")
  optparser.add_option("-n", "--numiter", dest="numiter", default=10, help="Num iterations")
  (opts, _) = optparser.parse_args()
  
  wordVecs = read_word_vecs(opts.input)
  lexicon = read_lexicon(opts.lexicon)
  numIter = int(opts.numiter)
  outFileName = opts.output
  
  print_word_vecs(retrofit(wordVecs, lexicon, numIter), outFileName)
  