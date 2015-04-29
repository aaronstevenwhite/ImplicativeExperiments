import os, sys, csv, argparse

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
## initialize parser
parser = argparse.ArgumentParser(description='Load data and run implicatives model.')

## file handling
parser.add_argument('--data', 
                    type=str, 
                    default='../data/results.filtered')
parser.add_argument('--output', 
                    type=str, 
                    default='./model')
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
## initialize parser
parser = argparse.ArgumentParser(description='Load data and run implicatives model.')

## file handling
parser.add_argument('--data', 
                    type=str, 
                    default='../data/results.ibex')
parser.add_argument('--mturk', 
                    type=str, 
                    default='../data/turk')
parser.add_argument('--dataout', 
                    type=str, 
                    default='../data/results.preprocessed')
parser.add_argument('--demoout', 
                    type=str, 
                    default='../data/demographics')


## parse arguments
args = parser.parse_args()


datafile = [line.strip() for line in open(args.data) if line[0] != '#']
subjlist = [line[-2].lower() for line in csv.reader(open(args.mturk))]

def process_line(linesplit):
    _, ident, _, _, _, cond, _, stimulus, response, _, rt  = linesplit

    itemlabel = stimulus.split()[1]

    verb, negation, followup, itemtype = cond.split('_')

    return ','.join([ident, itemlabel, verb, negation, followup, itemtype, response, rt])

with open(args.dataout, 'w') as dataout:
    with open(args.demoout, 'w') as demoout:

        dataout.write('subj,item,verb,frame,followup,itemtype,response,rt\n')
        demoout.write('subj,initials,age,education,location,language,otherlanguage,birthplace,sex,consent\n')

        demoinfo = []

        for line in datafile:
            linesplit = line.split(',')

            if linesplit[5] in ['consent', 'practice']:

                if not demoinfo:
                    demoinfo.append(linesplit[0])

                if linesplit[5] == 'consent':
                    demoinfo.append(linesplit[8].lower())

                if linesplit[7] == 'initials':
                    initials = linesplit[8].lower()

            elif initials in subjlist:
                outline = process_line(linesplit)
                dataout.write(outline+'\n')

                if demoinfo:
                    demoout.write(','.join(demoinfo)+'\n')

                demoinfo = []

            else:
                demoinfo = []
