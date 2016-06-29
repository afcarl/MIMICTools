import glob
import nltk
import os.path
import cPickle as pickle
import multiprocessing
from multiprocessing import Pool

import utils

#mimic_dir = '/data/ml2/jernite/MIMIC3/Parsed/MIMIC3_split'
mimic_dir = '/home/ankit/devel/data/MIMIC3_split'
mimic_dirs = sorted(glob.glob(mimic_dir + r'/[0-9]*'))
notes_files = [d + '/NOTEEVENTS_DATA_TABLE.csv' for d in mimic_dirs]
notes_files = [f for f in notes_files if os.path.isfile(f)]

def get_stats(notes_file):
    bigram_fd = nltk.FreqDist()
    trigram_fd = nltk.FreqDist()
    qgram_fd = nltk.FreqDist()
    for _, raw_text in utils.mimic_data([notes_file], super_verbose=True):
        sentences = nltk.sent_tokenize(raw_text)
        for sent in sentences:
            words = [w.lower() for w in nltk.word_tokenize(sent)]
            bigram_fd.update(nltk.ngrams(words, 2))
            trigram_fd.update(nltk.ngrams(words, 3))
            qgram_fd.update(nltk.ngrams(words, 4))
    return (bigram_fd, trigram_fd, qgram_fd)

p = Pool(int(.5 + (.9 * float(multiprocessing.cpu_count()))))
stats = p.map(get_stats, notes_files)

print 'Got the stats.'

bigram_fd = nltk.FreqDist()
trigram_fd = nltk.FreqDist()
qgram_fd = nltk.FreqDist()

for b, t, q in stats:
    bigram_fd.update(b)
    trigram_fd.update(t)
    qgram_fd.update(q)

print bigram_fd.most_common(50), '\n'
print trigram_fd.most_common(50), '\n'
print qgram_fd.most_common(50), '\n'

print 'Saving stats.'
with open('ngrams.pk', 'wb') as f:
    pickle.dump((bigram_fd, trigram_fd, qgram_fd), f, -1)
print 'Done.'
