import os, sys, csv

datafile = [line.strip() for line in open(sys.argv[1]) if line[0] != '#']
subjlist = [line[-2].lower() for line in csv.reader(open(sys.argv[2]))]

# 1. Time results were received.
# 2. MD5 hash of participant's IP address.
# 3. Controller name.
# 4. Item number.
# 5. Element number.
# 6. Type.
# 7. Group.
# 8. Question (NULL if none).
# 9. Answer.
# 10. Whether or not answer was correct (NULL if N/A).
# 11. Time taken to answer.

def process_line(linesplit):
    _, ident, _, _, _, cond, _, stimulus, response, _, rt  = linesplit

    itemlabel = stimulus.split()[1]

    verb, negation, followup, itemtype = cond.split('_')

    return ','.join([ident, itemlabel, verb, negation, followup, itemtype, response, rt])

with open(sys.argv[3], 'w') as outfile:

    outfile.write('subj,item,verb,negation,followup,itemtype,response,rt\n')

    for line in datafile:
        linesplit = line.split(',')

        if linesplit[5] in ['consent', 'practice']:
            if linesplit[7] == 'initials':
                initials = linesplit[8].lower()
            continue

        if initials in subjlist:
            outline = process_line(linesplit)
            outfile.write(outline+'\n')
