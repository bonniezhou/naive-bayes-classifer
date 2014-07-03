import nltk
import collections
import os
import json

def word_tally(rootdir):
    tokens = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            fullpath = os.path.join(root, file)
            f = open(fullpath, 'r')
            tokens = tokens + nltk.word_tokenize(f.read())

    tally = collections.Counter(tokens)
    email_type = rootdir
    newfile = open(rootdir + '_tally.json', 'w')
    json.dump(tally, newfile)
