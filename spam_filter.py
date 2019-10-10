from numpy import *
import sys

def loadDataSet(spamEmailNum, nonSpamEmailNum):
    
    emailList = [] # initialize the email list
    classVec = [] # initialize the class vector
    
    for  i in range(spamEmailNum): # load data of spam emails
        email = open("is_spam/%d.txt" % i, encoding = 'utf-8').read()
        email = email.split(' ')
        [word.lower() for word in email if len(word) > 3] # remove words less 
        emailList.append(email)                           # 4 characters, cast
        classVec.append(1)                                # into lowercase
        
    for  i in range(nonSpamEmailNum): # load data of non-spam emails
        email = open("not_spam/%d.txt" % i, encoding = 'utf-8').read()
        email = email.split(' ')
        [word.lower() for word in email if len(word) > 3] # remove words less 
        emailList.append(email)                           # 4 characters, cast
        classVec.append(0)                                # into lowercase
        
    return emailList, classVec

def vocabList(dataSet):
    """
    dataSet is a list of list of words that appear in emails
    """
    
    vocabSet = set([]) # initialize an empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) # add the vocabulary of each email 
    return list(vocabSet)                   # to the set, cast set to list

def setOfWordsVec(vocabulary, inputEmail):
    
    textVec = [0] * len(vocabulary) 
    for word in inputEmail:           
        if word in vocabulary:                     # find the occurrence of email
            textVec[vocabulary.index(word)] += 1   # words in vocabList
    return textVec                              

def trainNB(trainEmailVecList, trainClassVec):
    
    """
    trainEmailVecList is a list of vectors that shows the time of each word in 
    vocabList that shows up in the email.
    trainClassVec the vector the store wether each training email is spam or not
    """
    numTrainEmail = len(trainEmailVecList) # get the number of emails for training
    numWord = len(trainEmailVecList[0]) # get the length of the vocabList
    pSpam = sum(trainClassVec)/float(numTrainEmail) # P(email is span)
    p0Num = zeros(numWord) 
    p1Num = zeros(numWord)
    p0Denom = 0.0
    p1Denom = 0.0
    
    for i in range(numTrainEmail):
        if trainClassVec[i] == 1:
            p1Num += trainEmailVecList[i] # loads word occurrence in all spam emails
            p1Denom += sum(trainEmailVecList[i]) # loads total word number in email
        else:
            p0Num +=trainEmailVecList[i] # loads word occurrence in non-spam emails
            p0Denom += sum(trainEmailVecList[i]) # loads total word number in email
    
    p1Vec = p1Num / p1Denom
    p0Vec = p0Num / p0Denom
    return p0Vec, p1Vec, pSpam

def classifyNB(textVec, p0Vec, p1Vec, pSpam):
    
    p1 = sum(textVec * p1Vec) / (pSpam)
    p0 = sum(textVec * p0Vec) / (1 - pSpam)
    print('Probability of a spam email is: ', p1 / (p1 + p0))
    print('Probability of not a spam email is: ', p0 / (p1 + p0))
    if p1 > p0:
        return 1
    else:
        return 0
    
def test(emailName, spamEmailNum, nonSpamEmailNum):
    
    # load the email to test
    vocabulary = []
    emailContent = open(emailName, encoding = "utf-8").read()
    inputEmail = emailContent.split()
    [word.lower() for word in inputEmail if len(word) > 3]
    # get the vocabulary list
    trainEmailList, trainClassVec = loadDataSet(spamEmailNum, nonSpamEmailNum)
    vocabulary= vocabList(trainEmailList)

    trainEmailVecList = []
    for email in trainEmailList:
        trainEmailVecList.append(setOfWordsVec(vocabulary, email))
    testEmailVec = setOfWordsVec(vocabulary, inputEmail)
    p0Vec, p1Vec, pSpam = trainNB(trainEmailVecList, trainClassVec)
    
    testResult = classifyNB(testEmailVec, p0Vec, p1Vec, pSpam)
    if testResult == 1:
        print("%s has a greater probability to be spam email." % emailName)
    else:
        print("%s has a greater probability not to be spam email." % emailName)
    
    return testResult

def testCorrectness(spamEmailNum, nonSpamEmailNum):
    errorCount = 0
    for i in range(spamEmailNum):
        if test("is_spam/%d.txt" % i, spamEmailNum, nonSpamEmailNum) == 0:
            errorCount += 1
    for i in range(nonSpamEmailNum):
        if test("not_spam/%d.txt" % i, spamEmailNum, nonSpamEmailNum) == 1:
            errorCount += 1
    return errorCount / (spamEmailNum + nonSpamEmailNum)

def main():
    emailName = sys.argv[1]
    spamEmailNum = int(sys.argv[2])
    nonSpamEmailNum = int(sys.argv[3])
    testResult = test(emailName, spamEmailNum, nonSpamEmailNum)
    print(testCorrectness(spamEmailNum, nonSpamEmailNum))
    
if __name__ == "__main__":
    main()