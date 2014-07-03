import math
import nltk
import collections
import os


#tally training sets and store in .json file
def word_tally(training_set):
    tokens = []
    for root, dirs, files in os.walk(training_set):
        for file in files:
            fullpath = os.path.join(root, file)
            f = open(fullpath, 'r')
            tokens = tokens + nltk.word_tokenize(f.read())

    tally = collections.Counter(tokens)
    return tally


#naive bayes classfier
def nbc(ham_training, spam_training, email_path):   
    ham_dict = word_tally(ham_training)
    spam_dict = word_tally(spam_training)

    #calculate prior probabilities P(ham) and P(spam)
    ham_words = sum(ham_dict.values())
    spam_words = sum(spam_dict.values())
    all_words = ham_words + spam_words

    prior_ham = float(ham_words)/all_words
    prior_spam = float(spam_words)/all_words

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

    #total probabilities P(ham|email) and P(spam|email)
    total_ham_prob = ham_prob + math.log(prior_ham)
    total_spam_prob = spam_prob + math.log(prior_spam)

    if total_ham_prob > total_spam_prob:
        print "spam"
    else:
        print "ham"
