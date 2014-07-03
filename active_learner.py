import math
import nltk
import collections
import os

#--------------------------------------

ham_dict = {}
spam_dict = {}

number_of_rounds = 5

#--------------------------------------
#initialize ham_dict and spam_dict

initial_ham = "6"
initial_spam = "1117648339.27724_109.txt"

def tally(email):
    file = open(email, 'r')
    tokens = nltk.word_tokenize(file.read())
    tally = collections.Counter(tokens)
    return tally

ham_dict.update(tally(initial_ham))
spam_dict.update(tally(initial_spam))

#--------------------------------------

def email_pool_creator(training_set):
    email_pool = []
    for root, dirs, files in os.walk(training_set):
        for file in files:
            fullpath = os.path.join(root, file)
            f = open(fullpath, 'r')
            email_tokens = nltk.word_tokenize(f.read())
            email_dict = dict(collections.Counter(email_tokens))

            if 'ham' in fullpath:
                email_pool.append((email_dict, 'ham'))
            else:
                email_pool.append((email_dict, 'spam'))

    return email_pool

#-------------------------------------

training_pool = email_pool_creator("training_set")
test_pool = email_pool_creator("test_set")

#-------------------------------------

#determine the uncertainty ratio (ham v.s. spam) of a particular email
def uncertainty_ratio(email_dict):   

    #total number of words in ham and spam
    ham_words = sum(ham_dict.values())
    spam_words = sum(spam_dict.values())

    #calculate P(email|ham) and P(email|spam)
    ham_prob = 0
    spam_prob = 0

    if ham_words == 0 or spam_words == 0:
        return (1, 'ham') #skewed towards classification as 'ham'
        
    else:
        for word in email_dict:
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

    if ham_prob >= spam_prob: #skewed towards classification as 'ham'
        return (ratio, 'ham')
    elif ham_prob < spam_prob:
        return (ratio, 'spam')

    

#-------------------------------------

def sample(email_pool):
    rounds = 5
    highest_ratio = 0
    
    while rounds > 0:
        for email_tuple in email_pool:
            ratio = uncertainty_ratio(email_tuple[0])[0]
            if ratio > highest_ratio:
                highest_ratio = ratio
                email = email_tuple
		
		print email
        if email[1] == 'ham':
            ham_dict.update(email[0])
        else:
            spam_dict.update(email[0])

        del email_pool[email_pool.index(email)]
        rounds = rounds - 1


#---------------------------------------

def active_classifier(training_pool, test_pool):
    sample(training_pool)

    #[ham, spam]
    true_matrix = [0, 0]
    false_matrix = [0, 0]
    
    for email_tuple in test_pool:
        classification = uncertainty_ratio(email_tuple[0])[1]

        if classification == 'ham':
            if email_tuple[1] == 'ham':
                true_matrix[0] += 1
            else:
                false_matrix[0] += 1
        elif classification == 'spam':
            if email_tuple[1] == 'spam':
                true_matrix[1] += 1
            else:
                false_matrix[1] += 1

    return [true_matrix, false_matrix]


#active_learner.active_classifier(active_learner.training_pool, active_learner.test_pool)

