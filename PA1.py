#!Python3
#------------------------------------------------------------------------------
# Written by Vaan W. Waber
# Created on 1/8/2015
# Updated on 1/22/2015
# Description: This application creates Unigram and Bigram, both MLE and
# 	Katz Back-off, language models from training data in the form of a text file
#	of sentences delimited by newlines. These models are then used to generate
#	sentences, evaluate the probability of these sentences, and to determine
# 	the perplexity of the language models over test data.
#
# 	Created using Python 3.4. This application has only been tested on
#	Windows 8.1 x64, but should run on any machine with Python 3.4
#------------------------------------------------------------------------------

# Python built-in random number generator
import random
# Functions pack/unpack probabilities to avoid underflow
from math import log, pow
# Allows OS information to be gathered
import os

#------------------------------------------------------------------------------
# Main function used to keep program body at top of file,
# and to allow future use as a module.
def main():
#------------------------------------------------------------------------------

	userPrompt()
	
#------------------------------------------------------------------------------
# Provides a means for the user to interact with the application
# through use of the command prompt.
def userPrompt():
#------------------------------------------------------------------------------	

	bigram = False
	choice = False
	KB = False	

	while True:

		if os.name == 'nt':		
			os.system('CLS')
		else:
			os.system('clear')
			
		print('\nNLP Assignment 1, by Vaan Waber\n')
		print('Options: (Choose last option to populate more, up to 7)')
		print('  0: Exit')
		print('  1: Created processed corpus from text file')
		print('  2: Generate MLE bigram model in memory (includes MLE unigram)')
		
		if bigram:
			print('  3: Compute MLE unigram perplexity')
			print('  4: Print bigram model to file (for viewing)')
			print('  5: Generate Katz back-off bigram model')
			
			
		if bigram and KB:
			print('  6: Compute Katz back-off bigram perplexity')
			print('  7: Print generated sentences and respective probabilities to file')			
			
		print()		
		
		choice = input('Your Choice?: ')
		print('\n(Hit enter to use default value)\n')
		
		if choice == '0':
			break
			
		if choice == '1':			
		
			try:
				inFileName = input('Input file name? [pos_train.txt] : ') or 'pos_train.txt'
				outFileName = input('Output file name? [corpus.txt] : ') or 'corpus.txt'
				with open(inFileName, 'r') as inFile, open(outFileName, 'w+') as outFile:				
					preprocessCorpus(inFile, outFile, 1)
			except:
				print('\nInput file not found!')
				
		elif choice == '2':	
			try:
				corpusFileName = input('Corpus file name? [corpus.txt] : ') or 'corpus.txt'
				with open(corpusFileName, 'r') as corpusFile:
					bigram = bigramFromCorpus(corpusFile)
				generateMLEProbForBigram(bigram)
			except:
				print('\nCorpus file not found!')
				
		elif (choice == '3') and bigram:
			try:
				testFileName = input('Corpus file name? [pos_test.txt] : ') or 'pos_test.txt'
				with open(testFileName, 'r') as testFile:	
					perplexity = computeMLEUnigramPerplexityFromFile(bigram, testFile)
					print('\nPerplexity: ', perplexity)
			except:
				print('Test file not found!')
				
		elif (choice == '4') and bigram:
			try:
				bigramFileName = input('Output file name? [bigram.txt] : ') or 'bigram.txt'	
				with open(bigramFileName, 'w+') as bigramFile:
					printSortedBigramToFile(bigram, bigramFile)
			except:
				print('\nUnspecified error!')
			
		elif choice == '5':
			try:
				beta = input('Discount value? [0.5] : ') or 0.5
				beta = float(beta)
				generateKBProbForBigram(bigram, beta)
				KB = True
			except:
				print('\nInvalid value!')

		elif (choice == '6') and bigram and KB:
			try:
				testFileName = input('Corpus file name? [pos_test.txt] : ') or 'pos_test.txt'
				print('\nThis can take a couple minutes, please be patient\n')
				with open(testFileName, 'r') as testFile:	
					perplexity = computeKBBigramPerplexityFromFile(bigram, testFile)
					print('Perplexity: ', perplexity)
			except:
				print('Test file not found!')
			
		elif (choice == '7') and bigram and KB:
			
			try:
				sentenceFileName = input('Output file name? [sentences.txt] : ') or 'sentences.txt'
				count = input('Number of sentences per model(MLE Unigram and MLE Bigram)? [10] : ') or 10
				count = int(count)
				with open(sentenceFileName, 'w+') as sentenceFile:		
					printGeneratedSentencesAndProbsToFile(bigram, count, sentenceFile)
			except:
				print('\nInvalid value!')
				
		else:
			print('Invalid Choice!')
		
		input('\nDone! Press Enter to continue')

#------------------------------------------------------------------------------
# Accepts a bigram model pre-populated with KB probabilities and a 
# file open for reading containing test sentences delimited by newlines. 
# Returns the perplexity of the model over the test data. 
def computeKBBigramPerplexityFromFile(bigram, inFile):
#------------------------------------------------------------------------------

	wordCount = 0
	probSum = 0.0

	for line in inFile:
		sentence = line.strip().split()
		sentence.append('STOP')
		wordCount += len(sentence)
		
		#Function used to get probabilities for each sentence
		probSum += getKBBigramProbFromSentence(bigram, sentence)

	perplexity = pow(2, -(probSum/wordCount) )
	
	return(perplexity)
		
#------------------------------------------------------------------------------
# Accepts a bigram model, populated with MLE bigram probabilities
# and a file open for reading containing test sentences delimited by newlines. 
# Returns the perplexity of the model over the test data. 
def computeMLEUnigramPerplexityFromFile(bigram, inFile):
#------------------------------------------------------------------------------

	wordCount = 0
	probSum = 0.0

	for line in inFile:
		sentence = line.strip().split()
		sentence.append('STOP')
		wordCount += len(sentence)
		
		#Function used to get probabilities for each sentence
		probSum += getMLEUnigramProbFromSentence(bigram, sentence)

	perplexity = pow(2, -(probSum/wordCount) )
	
	return(perplexity)
			
#------------------------------------------------------------------------------
# Generates a number of sentences, computes the probability of each sentence,
# then prints to file. Unigram and Bigram MLE are used to generate sentences.
# Probabilities are generated from Unigram MLE and Bigram Katz Back-off.
def printGeneratedSentencesAndProbsToFile(bigram, count, outFile):
#------------------------------------------------------------------------------	
	
	outFile.write('-' * 80 + '\n')
	outFile.write('Unigram Generated Sentences:\n')
	outFile.write('-' * 80 + '\n\n')
	
	for i in range(count):		
		
		sentence = generateSentenceFromUnigram(bigram)
		
		probMLE = getMLEUnigramProbFromSentence(bigram, sentence)
		probMLE = unpackProb(probMLE)
		
		probKB = getKBBigramProbFromSentence(bigram, sentence)
		probKB = unpackProb(probKB)		
		
		outFile.write(' '.join(sentence))
		outFile.write('\n\n  Uni_MLE: {}\n  Bi_KB:   {}\n\n'.format(probMLE, probKB))
		
	outFile.write('-' * 80 + '\n')
	outFile.write('Bigram Generated Sentences:\n')
	outFile.write('-' * 80 + '\n\n')
		
	for i in range(count):
	
		sentence = generateSentenceFromBigram(bigram)
		
		probMLE = getMLEUnigramProbFromSentence(bigram, sentence)
		probMLE = unpackProb(probMLE)
		
		probKB = getKBBigramProbFromSentence(bigram, sentence)
		probKB = unpackProb(probKB)
		
		outFile.write(' '.join(sentence))
		outFile.write('\n\n  Uni_MLE: {}\n  Bi_KB:   {}\n\n'.format(probMLE, probKB))
		
#------------------------------------------------------------------------------
# Computes the MLE Unigram probability of a list of words. Expects a bigram
# data structure that is pre-populated with MLE Unigram probabilities.	
def getMLEUnigramProbFromSentence(bigram, sList):
#------------------------------------------------------------------------------	

	probability = 0.0	

	for word in sList:
	
		if word not in bigram:
			word = 'UNK'
		
		# Addition used instead of multiplication since probabilities are stored
		# stored internally using LOG
		probability += bigram[word][1]
		
	return(probability)

#------------------------------------------------------------------------------
# Computes the Katz Back-off Bigram probability of a list of words. Expects a 
# bigram data structure that is pre-populated with KB Bigram probabilities.	
# getKBProb() used to for probability computation.
def getKBBigramProbFromSentence(bigram, sList):
#------------------------------------------------------------------------------	
	
	probability = 0.0
	
	word1 = 'START'
	
	for word2 in sList:
	
		#Probability is summed instead of multiplied because it is packed using LOG
		#UNK words are handled in getKBProb()
		probability += getKBProb(bigram, word1, word2)		
		word1 = word2
		
	return(probability)		
	
#------------------------------------------------------------------------------
# This is a complex function that computes the Katz Back-off probabilities of
# a given word pair. Expects a bigram containing Unigram MLE probabilities, as
# well as discounted probabilities.
def getKBProb(bigram, word1, word2):
#------------------------------------------------------------------------------	
	
	probability = 0.0
	
	#If either word does not exist in the bigram it is treated as an unknown word.
	if word1 not in bigram:
		word1 = 'UNK'
		
	if word2 not in bigram:
		word2 = 'UNK'
	
	#If the word pair has been seen it training data, then that probability is used.
	if word2 in bigram[word1][2]:
		probability = bigram[word1][2][word2][2]
	#If the word pair has not been seen, then the Katz Back-off probability must be
	# computed
	else:
		alpha = 0.0
		
		
		for k, v in bigram[word1][2].items():
			alpha += ( unpackProb(v[2]) )
			
		alpha = 1.0 - alpha
		alpha = alpha * unpackProb(bigram[word2][1])
		mleB = 0.0
		
		for k, v in bigram.items():
			#Since START's MLE Prob is stored as 0 and 2^0=1, we have to discount START
			if (k != 'START') and (k not in bigram[word1][2]):
					mleB += unpackProb(v[1])
					
		alpha = (alpha / mleB)
		probability = packProb(alpha)	
	
	return(probability)		

#------------------------------------------------------------------------------
# Generates random list of words from MLE Bigram model using random number 
# generator
def generateSentenceFromBigram(bigram):
#------------------------------------------------------------------------------	

	lowest = ''
	sentence = []
	#Uses start symbol as default "last word"
	lastWord = 'START'
	
	
	while lowest != 'STOP':
	
		lowest = ''
		# Finds the max probability of all words following the last word
		# so the randomly generated probability can be normalized.
		maxProb = max(bigram[lastWord][2].items(), key=lambda e: e[1][1])[1][1]
		maxProb = unpackProb(maxProb)
		# Generates a random percentage of the max probability
		rand = packProb( maxProb * random.random() )
		
		#Finds the word with the lowest probability higher than the random percentage
		for k, v in bigram[lastWord][2].items():
		
			if (v[1]) > rand:
				if lowest:
					if v[1] < bigram[lowest][1]:
						lowest = k
				else:
					lowest = k
							
		sentence.append(lowest)
		lastWord = lowest
		
	return(sentence)
	
#------------------------------------------------------------------------------
# Generates random list of words from MLE Unigram model using random number 
# generator
def generateSentenceFromUnigram(bigram):
#------------------------------------------------------------------------------
	
	wordCount = wordCountBigram(bigram)	
	# Finds the probability of the word with the highest probability
	# so the randomly generated probability can be normalized.	
	maxCount = max(bigram.items(), key=lambda e: e[1][0])[1][0]			
	lowest = ''
	sentence = []
	
	while lowest != 'STOP':
	
		lowest = ''
		# Generates a random percentage of the max probability
		rand = packProb( random.random() * (maxCount/wordCount) )
	
		#Finds the word with the lowest probability higher than the random percentage
		for k, v in bigram.items():
				
			
			if (v[1]) > rand:
				if lowest:
					if v[1] < bigram[lowest][1]:
						lowest = k				
				else:
					lowest = k
					
		sentence.append(lowest)	
	
	return(sentence)
	
#------------------------------------------------------------------------------	
# Populates bigram with Katz Back-off probabilities for seen pairs of words
def generateKBProbForBigram(bigram, beta):
#------------------------------------------------------------------------------
		
	for k, v in bigram.items():
	
		for k2, v2 in v[2].items():
			
			#Probability is the count of the pair minus a give value
			# over the count of the first word
			v2[2] = packProb((v2[0]-beta)/v[0])
				
#------------------------------------------------------------------------------	
# Populates bigram with MLE Unigram and Bigram propabilities for seen words/pairs
def generateMLEProbForBigram(bigram):
#------------------------------------------------------------------------------

	wordCount = wordCountBigram(bigram)
			
	for k, v in bigram.items():
	
		if k != 'START':
			v[1] = packProb(v[0]/wordCount)

		for k2, v2 in v[2].items():
			
			v2[1] = packProb(v2[0]/v[0])
				
#------------------------------------------------------------------------------
# Counts total number of words seen in the bigram excluding start symobl
def wordCountBigram(bigram):
#------------------------------------------------------------------------------

	wordCount = 0
	for k, v in bigram.items():
		if k != 'START':
			wordCount += v[0]
		
	return(wordCount)
	
#------------------------------------------------------------------------------
# Creates bigram data structure from processed corpus file.
# Bigram stores all information for all models (MLE Unigram, MLE Bigram, KB Bigram)
def bigramFromCorpus(corpusFile):
#------------------------------------------------------------------------------

	# Dictionary(String, List(Int, Dictionary(String, Int))
	bigram = {}
	
	
	for line in corpusFile:
	
		line = removeTagsFromSentence(line)			
		line = line.split()
		
		for index, value in enumerate(line):

			word = line[index]		
			if (index+1) < len(line):
				nextWord = line[index+1]
			else:
				nextWord = ''
				
			if word in bigram:	
				bigram[word][0] += 1
				if nextWord:
					if nextWord in bigram[word][2]:						
						bigram[word][2][nextWord][0] += 1
					else:
						bigram[word][2][nextWord] = [1, 0, 0]
			else:
				if nextWord:
					bigram[word] = [1, 0, {nextWord: [1, 0, 0]}]
				else:
					bigram[word] = [1, 0, {}]
				
	#del bigram['START']
	return(bigram)
		
#------------------------------------------------------------------------------
# Prints bigram data structure to file for viewing, sorted based on word count
def printSortedBigramToFile(bigram, File):
#------------------------------------------------------------------------------
	
	for k, v in sorted(bigram.items(), key=lambda e: e[1][0], reverse=True):
		tmpStr = '"{0}", {1}, {2}:\n'.format( k, v[0], unpackProb(v[1]) )
		File.write(tmpStr)
		if v[2]:
			for k2, v2 in sorted(v[2].items(), key=lambda e: e[1][0], reverse=True):
				tmpStr2 = '     "{0}": {1}, MLE: {2}, KB: {3}\n'.format( k2, v2[0], unpackProb(v2[1]), unpackProb(v2[2]) )
				File.write(tmpStr2)	
		
#------------------------------------------------------------------------------
# Does initial formatting of the corpus
# Separates sentences using "<s>" tags
# Inserts start and stop symbols
# replaces words that occurs <= threshold with unknown symbol
def preprocessCorpus(inFile, outFile, threshold):
#------------------------------------------------------------------------------
	
	vocab = {}
	begin = '<s>'
	end = '</s>\n'
	start = 'START'
	stop = 'STOP'
	unk = 'UNK'
	
	for line in inFile:
		line = line.strip().split()
		for word in line:
			vocab[word] = vocab.get(word, 0) + 1
	
	inFile.seek(0)
	
	for line in inFile:
		line = line.strip().split()
		
		for index, word in enumerate(line):
			if vocab[word] <= threshold:
				line[index] = unk
				
		line.insert(0, start)
		line.append(stop)
		line.insert(0, begin)
		line.append(end)
		line = ' '.join(line)
		outFile.write(line)

#------------------------------------------------------------------------------
# Removes "<s>" tags from sentence string
def removeTagsFromSentence(sentence):
#------------------------------------------------------------------------------
	beg = '<s>'
	end = '</s>\n'
	#chops "beg" and "end" off of line
	return(sentence[len(beg):-len(end)])

#------------------------------------------------------------------------------
# Packages probabilities for internal use to prevent underflow
def packProb(unpackedProb):
#------------------------------------------------------------------------------

	packedprob = log(unpackedProb, 2)
	return(packedprob)

#------------------------------------------------------------------------------
# Unpacks propabilities packaged by packProb()
def unpackProb(packedProb):
#------------------------------------------------------------------------------

	unpackedProb = pow(2, packedProb)
	return(unpackedProb)
						
#------------------------------------------------------------------------------
# Main called at end of file so all function declarations happen before main
# body of program. If statement used to allow future use as a module.
if __name__ == "__main__":
	main()
#------------------------------------------------------------------------------	
		


		
	
	
