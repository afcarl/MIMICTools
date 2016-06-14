import nltk
from nltk.corpus import stopwords
import glob
import re
import cPickle as pickle
import os.path
from multiprocessing import Pool

#mimic_dir = '/data/ml2/jernite/MIMIC3/Parsed/MIMIC3_split'
mimic_dir = '/home/ankit/devel/data/MIMIC3_split'
mimic_dirs = sorted(glob.glob(mimic_dir + r'/[0-9]*'))
notes_files = [d + '/NOTEEVENTS_DATA_TABLE.csv' for d in mimic_dirs]
notes_files = [f for f in notes_files if os.path.isfile(f)]
print [(i, e) for i, e in enumerate(notes_files)]
print

re_anon = re.compile(r'ANON__.*?__ANON')

def treat_line(line):
    return re_anon.sub('<unk>', line)

def read_visit(lines):
    print lines[0]
    return treat_line(' '.join(lines[1:]))

stop = set(str(s) for s in stopwords.words('english'))
stop.add('unk')

def process_notes(notes_file):
    notes = 0
    print 'File', notes_file
    try:
        words = []
        with open(notes_file, 'r') as f:
            ct = 0
            st = []
            for line in f:
                ct += 1
                if ct % 50000 == 0:
                    print ct
                if line.strip() == '</VISIT>':
                    notes += 1
                    text = read_visit(st)
                    text = [w for s in nltk.sent_tokenize(text)
                                  for w in nltk.word_tokenize(s.lower())
                                      if w not in stop \
                                         and w.replace("'", '').isalpha()]
                    words += text
                    st = []
                elif line.strip() != '<VISIT>':
                    st += [line.strip()]
        fd = nltk.FreqDist(words)
        print 'Top 10:', fd.most_common(10)
        print 'Vocab size:', fd.B()
        print 'Notes count:', notes
        return (fd, notes)
    except IOError:
        return (None, None)

process_notes(notes_files[0])

#p = Pool(20)
#outs = p.map(process_notes, notes_files)
#fd = nltk.FreqDist()
#notes = 0
#for fdi, notesi in outs:
#    fd.update(fdi)
#    notes += notesi
#print
#print 'FINAL'
#print 'Top 10:', fd.most_common(10)
#print 'Vocab size:', fd.B()
#print 'Notes count:', notes
#print 'Writing FD ...',
#try:
#    with open('vocab_fd.pk', 'wb') as f:
#        pickle.dump(fd, f, -1)
#        print 'Done.'
#except IOError:
#    print 'Failed'
