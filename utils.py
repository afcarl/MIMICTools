import re

re_anon = re.compile(r'ANON__.*?__ANON')
re_sem = re.compile(r'\[\*\*.*?\*\*\]')


def subset(seq, k):
    if not 0<=k<=len(seq):
        for e in seq:
            yield e
    else:
        numbersPicked = 0
        for i,number in enumerate(seq):
            prob = (k-numbersPicked)/(len(seq)-i)
            if random.random() < prob:
                yield number
                numbersPicked += 1


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
    row_line = lines[0].split(',')
    return (row_line, treat_line(' '.join(lines[1:])))


def read_visit_sem(lines):
    row_line = lines[0].split('||||')
    return (row_line, treat_line_sem(' '.join(lines[1:])))


def mimic_data(notes_files, verbose=False):
    for notes_file in notes_files:
        if verbose:
            print 'MIMIC file', notes_file
        try:
            with open(notes_file, 'r') as f:
                ct = 0
                st = []
                nextst = []
                done = False
                for line in f:
                    ct += 1
                    #if ct % 100000 == 0:
                    #    print ct
                    if line.strip() == '</VISIT>' or done:
                        done = True
                        while done:
                            yield read_visit(st)
                            st = nextst
                            nextst = []
                            if line.strip() == '</VISIT>' and st:
                                done = True
                            else:
                                done = False
                    elif line.strip() != '<VISIT>':
                        content = line.strip()
                        if st and '"' in content:
                            nextcontent = content[:content.find('"')]
                            nextl = content[content.find('"')+1:].strip()
                            if nextl:
                                nextl = nextl.split(',', 9)
                                nextst = [','.join(nextl[:9])]
                                if nextl[9:]:
                                    nextst += nextl[9:]
                            done = True
                            st += [nextcontent]
                        elif not st:
                            content = content.split(',', 9)
                            st = [','.join(content[:9])]
                            if content[9:]:
                                st += content[9:]
                        else:
                            st += [content]
        except IOError:
            pass


def semeval_data(semeval_files, verbose=False):
    for semeval_file in semeval_files:
        if verbose:
            print 'SemEval file', semeval_file
        with open(semeval_file, 'r') as f:
            yield read_visit_sem([l.strip() for l in f])
