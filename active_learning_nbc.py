import math
import nltk
import collections
import os


#tally training sets and store in dictionary
def word_tally(training_set):
    tokens = []
    for root, dirs, files in os.walk(training_set):
        for file in files:
            fullpath = os.path.join(root, file)
            f = open(fullpath, 'r')
            tokens = tokens + nltk.word_tokenize(f.read())

    tally = collections.Counter(tokens)
    return tally

#--------------------------------------

#Constants
ham_dict = word_tally("training_ham")
spam_dict = word_tally("training_spam")

#--------------------------------------

#naive bayes classifier (suming prior probs of ham & spam are equal)
#Example: nbc(ham_dict, spam_dict, "6") => 'ham'
#Example: bc(ham_dict, spam_dict, "1117648339.27724_109.txt") => 'spam'

def nbc(ham_dict, spam_dict, email_path):

    #total number of words in ham and spam
    ham_words = sum(ham_dict.values())
    spam_words = sum(spam_dict.values())

    #calculate P(email|ham) and P(email|spam)
    email_file = open(email_path, 'r')
    email_tokens = nltk.word_tokenize(email_file.read())

    ham_prob = 0
    spam_prob = 0
    for word in email_tokens:
        if word in ham_dict:
            prob_word_ham = ham_dict[word] / float(ham_words)
            logp_word_ham = math.log(prob_word_ham)
            ham_prob = ham_prob + logp_word_ham

        if word in spam_dict:
            prob_word_spam = spam_dict[word] / float(spam_words)
            logp_word_spam = math.log(prob_word_spam)
            spam_prob = spam_prob + logp_word_spam

    ham_prob = abs(ham_prob)
    spam_prob = abs(spam_prob)

    if ham_prob > spam_prob:
        return 'ham'
    else:
        return 'spam'

#-------------------------------------

#determine the uncertainty ratio (ham v.s. spam) of a particular email
def uncertainty_ratio(ham_dict, spam_dict, email_path):   

    #total number of words in ham and spam
    ham_words = sum(ham_dict.values())
    spam_words = sum(spam_dict.values())

    #calculate P(email|ham) and P(email|spam)
    email_file = open(email_path, 'r')
    email_tokens = nltk.word_tokenize(email_file.read())

    ham_prob = 0
    spam_prob = 0
    for word in email_tokens:
        if word in ham_dict:
            prob_word_ham = ham_dict[word] / float(ham_words)
            logp_word_ham = math.log(prob_word_ham)
            ham_prob = ham_prob + logp_word_ham

        if word in spam_dict:
            prob_word_spam = spam_dict[word] / float(spam_words)
            logp_word_spam = math.log(prob_word_spam)
            spam_prob = spam_prob + logp_word_spam

    ham_prob = abs(ham_prob)
    spam_prob = abs(spam_prob)
    
    #assuming prior probabilities are equal, calculate the uncertainty between P(email|ham) and P(email|spam)
    ratio = min([ham_prob, spam_prob]) / float(max([ham_prob, spam_prob]))
    return ratio

#-------------------------------------

#find the email with the highest uncertainty ratio
#must be in the same folder as the training sets (named "training_ham" and "training_spam")
#Example: sample("unclassified_pool", "highest") => 'unclassified_pool\\test_spam\\2004\\10\\1096866532.15172_93.txt'
def sample(email_pool, uncertainty):

    if uncertainty == "highest":
        highest_ratio = 0
    elif uncertainty == "lowest":
        lowest_ratio = 1
    else:
        return "invalid uncertainty superlative"
    email = 0

    #total number of words in ham and spam
    ham_words = sum(ham_dict.values())
    spam_words = sum(spam_dict.values())
    
    #calculate uncertainty_ratio for each email in email_pool, update highest_ratio
    for root, dirs, files in os.walk(email_pool):
        for file in files:
            fullpath = os.path.join(root, file)
            ratio = uncertainty_ratio(ham_dict, spam_dict, fullpath)
            
            if uncertainty == "highest":
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    email = fullpath
            elif uncertainty == "lowest":
                if ratio < lowest_ratio:
                    lowest_ratio = ratio
                    email = fullpath        
                
    return email

#---------------------------------------

#active learning implementation (sampling the highest uncertainty)
def active_learning(email_pool):

    user_continue = True

    while user_continue:
        email_sample = sample(email_pool, "highest")
        print email_sample
        f = open(email_sample, 'r')
        email_tokens = nltk.word_tokenize(f.read())
        email_dict = collections.Counter(email_tokens)
        oracle_input = raw_input("Is this ham or spam? (Type 'quit' to exit.) ")

        if oracle_input == "ham":
            ham_dict.update(email_dict)
        elif oracle_input == "spam":
            spam_dict.update(email_dict)
        elif oracle_input == "quit":
            user_continue = False
        else:
            print "invalid classification"

        #remove email from email_pool

#--------------------------------------

#active learning implementation (sampling the most uncertain, then adding that and the least uncertain)
def active_learning_switch(email_pool):

    user_continue = True

    while user_continue:
        email_sample = sample(email_pool, "highest")
        print email_sample
        f = open(email_sample, 'r')
        email_tokens = nltk.word_tokenize(f.read())
        email_dict = collections.Counter(email_tokens)
        oracle_input = raw_input("Is this ham or spam? (Type 'quit' to exit.) ")

        if oracle_input == "ham":
            ham_dict.update(email_dict)
        elif oracle_input == "spam":
            spam_dict.update(email_dict)
        elif oracle_input == "quit":
            user_continue = False
            break
        else:
            print "invalid classification"
            break

        certain_email = sample(email_pool, "lowest")
        f = open(certain_email, 'r')
        certain_email_dict = collections.Counter(nltk.word_tokenize(f.read()))
        certain_classification = nbc(ham_dict, spam_dict, certain_email)
        print certain_email, certain_classification
        if certain_classification == 'ham':
            ham_dict.update(certain_email_dict)
        else:
            spam_dict.update(certain_email_dict)

        #remove email_sample and certain_email from email_pool
        
