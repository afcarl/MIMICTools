import nltk
from nltk.corpus import stopwords
import glob
import re
import cPickle as pickle
import os.path
import multiprocessing
import difflib
from multiprocessing import Pool

mimic_dir = '/data/ml2/jernite/MIMIC3/Parsed/MIMIC3_split'
#mimic_dir = '/home/ankit/devel/data/MIMIC3_split'
mimic_dirs = sorted(glob.glob(mimic_dir + r'/[0-9]*'))
notes_files = [d + '/NOTEEVENTS_DATA_TABLE.csv' for d in mimic_dirs]
notes_files = [f for f in notes_files if os.path.isfile(f)]

semeval_dir = '/data/ml2/ankit/semeval_data'
#semeval_dir = '/home/ankit/devel/data/data'
semeval_dirs = glob.glob(semeval_dir + r'/*')
semeval_files = sum([glob.glob(d + r'/*.text') for d in semeval_dirs], [])

re_anon = re.compile(r'ANON__.*?__ANON')
re_sem = re.compile(r'\[\*\*.*?\*\*\]')

def treat_line(line):
    if 'ANON__' in line:
        return re_anon.sub('<unk>', line)
    else:
        return line

def treat_line_sem(line):
    if '[**' in line:
        return re_sem.sub('<unk>', line)
    else:
        return line

def read_visit(lines, subject_id):
    #"ROW_ID","RECORD_ID",*"SUBJECT_ID"*,"HADM_ID","CHARTDATE","CATEGORY",
    #"DESCRIPTION","CGID","ISERROR","TEXT"
    row_line = lines[0].split(',') #SUBJECT_ID matches col 2 of semeval.
    subid = int(row_line[2])
    if subid != subject_id:
        return (None, None)
    row = int(row_line[0])
    return (row, treat_line(' '.join(lines[1:])))

def read_visit_sem(lines):
    row_line = lines[0].split('||||')
    subid = int(row_line[1])
    return (subid, treat_line_sem(' '.join(lines[1:])))

stop = set(str(s) for s in stopwords.words('english'))
stop.add('unk')

with open('vocab_fd.pk', 'rb') as f:
    fd = pickle.load(f)
print 'Before removing hapaxes:', fd.B()
#haps = fd.hapaxes()
#for h in haps:
#    del fd[h]
#print 'After removing hapaxes:', fd.B()
vocab = sorted(fd.keys())
print 'Vocab loaded'

def find_semeval(semeval_file):
    '''Returns a (mimic_file, row_idx, thres)
    '''
    print 'SemEval file', semeval_file
    try:
        with open(semeval_file, 'r') as f:
            st = []
            for line in f:
                st += [line.strip()]
            subject_id, semeval_text = read_visit_sem(st)
            semeval_text = [w for s in nltk.sent_tokenize(semeval_text)
                                  for w in nltk.word_tokenize(s.lower())
                                      if w in fd]
    except IOError:
        return (None, None)
    for thres in [1., .95, .9, .85, .8, .75, .7, .65, .6, .55, .5]:
        print 'Threshold', thres
        for notes_file in notes_files:
            print 'File', notes_file
            try:
                with open(notes_file, 'r') as f:
                    ct = 0
                    st = []
                    done = False
                    for line in f:
                        ct += 1
                        if ct % 100000 == 0:
                            print ct
                        if line.strip() == '</VISIT>' or done:
                            row, text = read_visit(st, subject_id)
                            if text:
                                text = [w for s in nltk.sent_tokenize(text)
                                        for w in nltk.word_tokenize(s.lower())
                                        if w in fd]
                                if difflib.SequenceMatcher(a=semeval_text,
                                              b=text,
                                              autojunk=False).ratio() >= thres:
                                    return (notes_file, row, thres)
                            st = nextst
                            nextst = []
                            done = False
                        elif line.strip() != '<VISIT>':
                            content = line.strip()
                            if st and '"' in content:
                                nextcontent = content[:content.find('"')]
                                nextst = [content[content.find('"'):]]
                                done = True
                                content = nextcontent
                            st += [content]
            except IOError:
                pass
    return (None, None)

p = Pool(int(.5 + (.9 * float(multiprocessing.cpu_count()))))
outs = p.map(find_semeval, semeval_files)

output = zip(semeval_files, outs)
print 'Saving output'
with open('matches.pk', 'wb') as f:
    pickle.dump(output, f, -1)
for f, o in output:
    print f, o
