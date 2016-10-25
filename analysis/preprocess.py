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
                    default='../data/results.ibex')
parser.add_argument('--prompt', 
                    type=str, 
                    default='entailment')
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
subjlist = [line[15].strip('"').lower() for line in csv.reader(open(args.mturk))]

def process_line(linesplit):
    _, ident, _, itemnum, _, cond, _, stimulus, response, _, rt  = linesplit

    itemlabel = stimulus.split()[1] if args.prompt != 'likelihood' else stimulus.split()[6]

    if args.prompt == 'entailment':
        verb, negation, followup, itemtype = cond.split('_')
        return ','.join([ident, itemlabel, itemnum, verb, negation, followup, itemtype, response, rt])
    else:
        cond_list = cond.split('_')

        if len(cond_list) == 3:
            verb, negation, itemtype = cond_list
        elif len(cond_list) == 4:
            verb, negation, itemtype, _ = cond_list

        return ','.join([ident, itemlabel, itemnum, verb, negation, itemtype, response, rt])

with open(args.dataout, 'w') as dataout:
    with open(args.demoout, 'w') as demoout:

        if args.prompt == 'entailment':
            dataout.write('subj,context,item,verb,frame,followup,itemtype,response,rt\n')
        else:
            dataout.write('subj,context,item,verb,frame,itemtype,response,rt\n')

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
                    
            elif initials in subjlist or args.prompt=='entailment':
                if len(linesplit) == 11:
                    outline = process_line(linesplit)
                    dataout.write(outline+'\n')

                if demoinfo:
                    demoout.write(','.join(demoinfo)+'\n')

                demoinfo = []

            else:
                demoinfo = []
