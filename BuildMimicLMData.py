import glob
import nltk
import os.path
import cPickle as pickle
import re
from os.path import join as pjoin
import multiprocessing
from multiprocessing import Pool

from mimictools import utils as mutils

vocab_size = 25000-2

mimic_dir = '/data/ml2/jernite/MIMIC3/Parsed/MIMIC3_split'
out_dir = '/data/ml2/ankit/MIMICTools/output'

EOS = '+'
UNK = '|'

fix_re = re.compile(r"[^a-z0-9/'?.,-]+")
num_re = re.compile(r'[0-9]+')
dash_re = re.compile(r'-+')

def fix_word(word):
    word = word.lower()
    word = fix_re.sub('-', word).strip('-')
    word = num_re.sub('#', word)
    word = dash_re.sub('-', word)
    return word


print 'Loading vocab FD'
with open('vocab_fd.pk', 'rb') as f:
    vocab_fd = pickle.load(f)
vocab_list = [k for k,v in vocab_fd.most_common(vocab_size)]
vocab_list.insert(0, EOS) # end of sentence
vocab_list.insert(1, UNK) # unknown
print 'Saving truncated vocab'
with open(pjoin(out_dir, 'vocab.pk'), 'wb') as f:
    pickle.dump(vocab_list, f, -1)
vocab_set = set(vocab_list)
vocab_lookup = {word: idx for (idx, word) in enumerate(vocab_list)}

print 'Processing notes ...'


def prepare_dataset(split):
    notes_file = pjoin(mimic_dir, '%02d/NOTEEVENTS_DATA_TABLE.csv' % (split,))
    if os.path.isfile(notes_file):
        print 'Starting split', split
        raw_data = []
        for _, raw_text in mutils.mimic_data([notes_file], replace_anon='_'):
            sentences = nltk.sent_tokenize(raw_text)
            for sent in sentences:
                words = [fix_word(w) for w in nltk.word_tokenize(sent)
                                        if any(c.isalpha() or c.isdigit()
                                            for c in w)]
                finalwords = []
                for word in words:
                    if not word: continue
                    if word in vocab_set:
                        finalwords.append(vocab_lookup[word])
                    else:
                        finalwords.append(vocab_lookup[UNK])
                finalwords.append(vocab_lookup[EOS])
                raw_data.extend(finalwords)
        with open(pjoin(out_dir, 'notes_%02d.pk' % (split,)), 'wb') as f:
            pickle.dump(raw_data, f, -1)
            print 'Wrote split', split


p = Pool(int(.5 + (.9 * float(multiprocessing.cpu_count()))))
p.map(prepare_dataset, range(100))
