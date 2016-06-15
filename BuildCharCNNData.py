import glob
import nltk
import os.path
import random
from multiprocessing import Pool

import utils

train_size = 20
valid_size = 3
test_size = 3

mimic_dir = '/data/ml2/jernite/MIMIC3/Parsed/MIMIC3_split'
#mimic_dir = '/home/ankit/devel/data/MIMIC3_split'
mimic_dirs = sorted(glob.glob(mimic_dir + r'/[0-9]*'))
notes_files = [d + '/NOTEEVENTS_DATA_TABLE.csv' for d in mimic_dirs]
notes_files = [f for f in utils.subset([f for f in notes_files
                                              if os.path.isfile(f)],
                                        train_size + valid_size + test_size)]
random.shuffle(notes_files)

files = {}
files['train'] = notes_files[:train_size]
notes_files = notes_files[train_size:]
files['valid'] = notes_files[:valid_size]
files['test'] = notes_files[valid_size:]

def prepare_dataset(out):
    with open(out + '.txt', 'w') as f:
        for _, raw_text in utils.mimic_data(files[out], replace_anon='*unk*',
                                            verbose=True):
            sentences = nltk.sent_tokenize(raw_text)
            for sent in sentences:
                words = [w.lower() for w in nltk.word_tokenize(sent)
                                       if any(c.isalpha() or c.isdigit()
                                              for c in w)]
                line = ' '.join(words)
                if line:
                    print >> f, ' ' + line.replace('*unk*', '<unk>')

p = Pool(3)
p.map(prepare_dataset, ['train', 'valid', 'test'])
