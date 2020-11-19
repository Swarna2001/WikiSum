import nltk

'''
word_tokenize - function to tokenize a given sentence into a list of words
sent_tokenize - function to tokenize a given paragraph into a list of sentences
stopwords - function to generate the list of stopwords pertaining to a given language
PorterStemmer - algorithm to find out the root stem of a given word
string - a class used to remove punctuations from the given text
'''
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import csv
import numpy as np
import wikipedia

class neural_network(object):
  def __init__(self):
  #parameters
    self.inputSize = 6
    self.outputSize = 1
    self.hiddenSize = 12

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (10x20) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (20x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 10x20 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 20x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  """def predict(self):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(x_predicted))
    print("Output: \n" + str(self.forward(x_predicted)))"""


def preprocess(givenFile):
    '''
    This function is used for preprocessing the data contained in the 
    supplied file object givenFile argument. It tokenises the given 
    data into a set of sentences first, followed by word level as well.
    '''
    
    '''Returns a list consisting of the sentences of the article'''
    webPageSentences = sent_tokenize(givenFile)
    
    '''Returns a list consisting of the sentences of the article without punctuations'''
    webPageRemovedPunctuations = ["".join([char for char in s if char not in string.punctuation]) for s in webPageSentences]
    
    '''Returns a list consisting of the sentences of the article, where each sentence is now a list consisting 
        of the corresponding words that make it up'''
    webPageWords = [word_tokenize(n) for n in webPageRemovedPunctuations]
    
    '''Stop words are removed'''
    stop_words = stopwords.words("english")
    webPageFilteredWords = [[word for word in s if word not in stop_words] for s in webPageWords]
    
    '''Each word present is converted to its root stem using PorterStemmer Algorithm'''
    porter = PorterStemmer()
    webPageStemmed = [[porter.stem(word) for word in s] for s in webPageFilteredWords]
    
    return webPageStemmed


def feature_vector(preProcessedContent,sentence,sentenceSimilarity,):
    ''' This function will return a list of 10 elements corresponding to the
    value of the 10 features.'''
    
    '''f1 = Sentence position.We assume that the first sentences of a paragraph are the most important. 
    Therefore, we rank a paragraph sentence according to its position in the paragraph and we consider 
    maximum positions of 5.'''
    positionOfSentence = preProcessedContent.index(sentence)
    f1 = 0
    if positionOfSentence in range(0, 5):
        f1 = (5 - positionOfSentence) / 5
    else:
        f1 = 0

    count = 0
    tagged = nltk.pos_tag(sentence)
    for i in range(len(tagged)):
        if(tagged[i][1]) == 'NNP':
            count = count + 1
    f2 = count/len(tagged)
    ''' f3 = similarites with other sentences (Bushy method)'''
    f3 = 0
    for i in sentenceSimilarity[positionOfSentence]:
        if i > 0.1:
            f3 += 1
        
    ''' f4 = Sentence centrality (similarity with rest of document).Sentence centrality is the vocabulary overlap
    between this sentence and other sentences in the document.'''
    distinctWordsInSent = set(sentence)
    f4 = len(distinctWordsInSent.intersection(bagOfWords)) / len(distinctWordsInSent.union(bagOfWords))
    
    ''' f5 = similarites with other sentences( aggregate similarities) '''
    f5 = 0
    for i in sentenceSimilarity[positionOfSentence]:
        f5 += i
    
    '''f7 = sentence relative length.This feature is employed to penalize sentences that are too short, since these
    sentences are not expected to belong to the summary.'''
    f7 = len(sentence) * averageSentLength    
    
    return [f1, f2,f3,f4, f5, f7]

if __name__ == "__main__":


    f = open("para.txt",'r')
    content = f.read()
    print(type(content))
    preProcessedContent = preprocess(content)


    averageSentLength = 0

    '''Finding out the vocabulary of the document'''
    bagOfWords = []
    for i in preProcessedContent:
        bagOfWords.extend(i)
        averageSentLength += len(i)
        
    bagOfWords = set(bagOfWords)

    sentenceSimilarity = []
    for i in range(len(preProcessedContent)):
        temporarySimilarity = []
        for j in range(len(preProcessedContent)):
            if i == j :
                continue
            numOfSimilarWords = len(set(preProcessedContent[i]).difference(set(preProcessedContent[j])))
            temporarySimilarity.append(numOfSimilarWords / max(len(preProcessedContent[i]), len(preProcessedContent[j])))
        sentenceSimilarity.append(temporarySimilarity)

    featureVector = []
    value = []
    for s in preProcessedContent:
        vector = feature_vector(preProcessedContent,s,sentenceSimilarity)
        featureVector.append(vector)
        value.append(sum(vector))
    print(featureVector) 
    values = []
    
    nn = neural_network()
    for i in range(len(featureVector)): # trains the nn with all the sentences in one document
        print("Input (scaled): \n", np.array(featureVector[i]))
        #print("Actual Output: \n", np.array(actualValues[i]))
        values.append (str(nn.forward(featureVector[i])))
        

    print(values)  
    con = content.split('.')
    l = len(value) 
    print(con)
    sent = []
    max = np.max(value)
    for j in range(l//2):
        for i in range(len(value)):
            if value[i] == max:
                sent.append(i)
                value[i]=-1
                max = np.max(value)
                break
    print(sent)
    for i in sent:
        print(con[i],".",end="\n")