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
patients_dir = '/data/ml2/ankit/MIMIC3pk'
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
with open('vocab.list', 'w') as f: # for CharCNN
    for w in vocab_list:
        print >> f, w
vocab_set = set(vocab_list)
vocab_lookup = {word: idx for (idx, word) in enumerate(vocab_list)}

print 'Loading aux vocab'
with open(pjoin(patients_dir, 'vocab_aux.pk'), 'rb') as f:
    # loads labs, diagnoses, procedures, prescriptions
    aux_list = pickle.load(f)
aux_list['admission_type'] = ['ELECTIVE', 'URGENT', 'NEWBORN', 'EMERGENCY']
aux_set = {feat: set(vals) for (feat, vals) in aux_list.items()}
aux_lookup = {feat: {val: idx for (idx, val) in enumerate(vals)}
                    for (feat, vals) in aux_list.items()}

print 'Processing notes ...'

def prepare_dataset(split):
    notes_file = pjoin(mimic_dir, '%02d/NOTEEVENTS_DATA_TABLE.csv' % (split,))
    if os.path.isfile(notes_file):
        print 'Starting split', split
        with open(pjoin(patients_dir, 'patients_%02d.pk' % (split,))) as f:
            patients = pickle.load(f)
        raw_data = []
        for row_line, raw_text in mutils.mimic_data([notes_file], replace_anon='_'):
            note_data = []
            try:
                subject_id = int(row_line[2])
            except ValueError:
                subject_id = row_line[2]
            try:
                admission_id = int(row_line[3])
            except ValueError:
                admission_id = row_line[3]
            try:
                patient = patients[subject_id]
                admission = patient['ADMISSIONS'][admission_id]
            except KeyError:
                # XXX workaround to the splitting data bug that starts new entries on same lines as prev entry ends.
                continue

            try:
                gender = 1 if patient['GENDER'] == 'F' else 0
                has_dod = 1 if patient['DOD'] else 0
                has_icu_stay = 1 if admission.get('ICU_STAYS', []) else 0
                admission_type = aux_lookup['admission_type'][admission['ADMISSION_TYPE']]
                diagnoses = [aux_lookup['diagnoses'][d['ICD9_CODE']] for d in admission.get('DIAGNOSES', [])]
                procedures = [aux_lookup['procedures'][d['ICD9_CODE']] for d in admission.get('PROCEDURES', [])]
                labs = [aux_lookup['labs'][d['ITEMID']] for d in admission.get('LABS', [])]
                prescriptions = [aux_lookup['prescriptions'][d['NDC']] for d in admission.get('PRESCRIPTIONS', [])]
            except KeyError:
                continue

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
                note_data.extend(finalwords)
            raw_data.append((note_data, gender, has_dod, has_icu_stay, admission_type, diagnoses, procedures, labs, prescriptions))
        with open(pjoin(out_dir, 'notes_%02d.pk' % (split,)), 'wb') as f:
            pickle.dump(raw_data, f, -1)
            print 'Wrote split', split


p = Pool(int(.5 + (.9 * float(multiprocessing.cpu_count()))))
p.map(prepare_dataset, range(100))
