from string import Template
import random

all_items = []

verbs = {'want' : 'wanted', 
         'hope' : 'hoped',
         'remember' : 'remembered',
         'forget' : 'forgot',
         'manage' : 'managed',
         'fail' : 'failed',
         'know' : 'knew',
         'think' : 'thought',
         'refuse' : 'refused',
         'neglect' : 'neglected',
         'mean' : 'meant',
         'hesitate' : 'hesitated',
         'opt' : 'opted',
         'hasten' : 'hastened'
         }

predicates = {'talk' : ('The customer', 
                        'the employee', 
                        'The employee', 
                        'talk to the manager', 
                        'talked to the manager', 
                        'talked to the manager', 
                        'talking to the manager'),
              'close' : ('The businessperson', 
                         'her colleague', 
                         "The businessperson's colleague", 
                         'close the sales deal', 
                         'closed the sales deal', 
                         'closed the sales deal', 
                         'closing the sales deal'),
              'take' : ('The student', 
                        'her friend', 
                        "The student's friend", 
                        "take the professor's class", 
                        "took the professor's class", 
                        "taken the professor's class", 
                        "taking the professor's class"),
              'return' : ('The author', 
                          'her agent', 
                          "The author's agent", 
                          "return the publisher's call", 
                          "returned the publisher's call", 
                          "returned the publisher's call", 
                          "returning the publisher's call")
              }

control_frames = {'positive' : lambda verb, pred_tuple: "{} {} to {}.".format(pred_tuple[0], verb, pred_tuple[3]),
                  'negative' : lambda verb, pred_tuple: "{} didn't {} to {}.".format(pred_tuple[0], verb, pred_tuple[3]),
                  'embhighneg' : lambda verb, pred_tuple: "{} {} not to {}.".format(pred_tuple[0], verb, pred_tuple[3]),
                  'emblowneg' : lambda verb, pred_tuple: "{} {} to not {}.".format(pred_tuple[0], verb, pred_tuple[3])}

hadto_frames = {'positive' : lambda verb, pred_tuple: "{} {} that she had to {}.".format(pred_tuple[0], verb, pred_tuple[3]),
                 'negative' : lambda verb, pred_tuple: "{} didn't {} that she had to {}.".format(pred_tuple[0], verb, pred_tuple[3]),
                 'embhighneg' : lambda verb, pred_tuple: "{} {} that she didn't have to {}.".format(pred_tuple[0], verb, pred_tuple[3]),
                 'emblowneg' : lambda verb, pred_tuple: "{} {} that she had to not {}.".format(pred_tuple[0], verb, pred_tuple[3])}

ecm_frames = {'positive' : lambda verb, pred_tuple: "{} {} {} to {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[3]),
              'negative' : lambda verb, pred_tuple: "{} didn't {} {} to {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[3]),
              'embhighneg' : lambda verb, pred_tuple: "{} {} {} not to {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[3]),
              'emblowneg' : lambda verb, pred_tuple: "{} {} {} to not {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[3])}

forto_frames = {'positive' : lambda verb, pred_tuple: "{} {} for {} to {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[3]),
                'negative' : lambda verb, pred_tuple: "{} didn't {} for {} to {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[3]),
                'embhighneg' : lambda verb, pred_tuple: "{} {} for {} not to {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[3]),
                'emblowneg' : lambda verb, pred_tuple: "{} {} for {} to not {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[3])}

finite_frames = {'positive' : lambda verb, pred_tuple: "{} {} that {} had {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5]),
                 'negative' : lambda verb, pred_tuple: "{} didn't {} that {} had {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5]),
                 'embhighneg' : lambda verb, pred_tuple: "{} {} that {} hadn't {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5]),
                 'emblowneg' : lambda verb, pred_tuple: "{} {} that {} hadn't {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5])}

question_frames = {'positive' : lambda verb, pred_tuple: "{} {} whether {} had {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5]),
                   'negative' : lambda verb, pred_tuple: "{} didn't {} whether {} had {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5]),
                   'embhighneg' : lambda verb, pred_tuple: "{} {} whether or not {} had {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5]),
                   'emblowneg' : lambda verb, pred_tuple: "{} {} whether {} hadn't {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5])}

about_frames = {'positive' : lambda verb, pred_tuple: "{} {} about {}.".format(pred_tuple[0], verb, pred_tuple[6]),
                'negative' : lambda verb, pred_tuple: "{} didn't {} about {}.".format(pred_tuple[0], verb, pred_tuple[6]),
                'embhighneg' : lambda verb, pred_tuple: "{} {} about not {}.".format(pred_tuple[0], verb, pred_tuple[6]),
                'emblowneg' : lambda verb, pred_tuple: "{} {} about not {}.".format(pred_tuple[0], verb, pred_tuple[6])}

tosay_frames = {'positive' : lambda verb, pred_tuple: "{} {} to say that {} had {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5]),
                 'negative' : lambda verb, pred_tuple: "{} didn't {} to say that {} had {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5]),
                 'embhighneg' : lambda verb, pred_tuple: "{} {} to say that {} hadn't {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5]),
                 'emblowneg' : lambda verb, pred_tuple: "{} {} to say that {} hadn't {}.".format(pred_tuple[0], verb, pred_tuple[1], pred_tuple[5])}

hadto_verbs = ['hope', 'remember', 'forget', 'know', 'think']

frames_fillers = {'want'     : {'forto' : forto_frames, 'ecm' : ecm_frames},
                  'hope'     : {'finite' : finite_frames, 'forto' : forto_frames},
                  'remember' : {'finite' : finite_frames, 'question' : question_frames},
                  'forget'   : {'finite' : finite_frames, 'question' : question_frames},
                  #'manage'   : None,
                  #'fail'     : None,
                  'know'     : {'finite' : finite_frames, 'question' : question_frames},
                  'think'    : {'finite' : finite_frames},
                  #'refuse'   : None,
                  'neglect'  : {'tosay' : tosay_frames},
                  'mean'     : {'tosay' : tosay_frames},
                  'hesitate' : {'about' : about_frames},
                  #'opt'      : None,
                  #'hasten'   : None
              }

filler_tuples = [(verb_base, filler_type) for verb_base, filler_dict in frames_fillers.iteritems() for filler_type in filler_dict]

follow_up_test = lambda pred_tuple: "Did {} {}?".format(pred_tuple[0].lower(), pred_tuple[3])
follow_up_filler = lambda pred_tuple: "Did {} {}?".format(pred_tuple[2].lower(), pred_tuple[3])

item_temp = lambda typ, first, second: Template('''[$typ, "Question", {q: "$first <br /><br /> $second", as: ["Yes", "Maybe or maybe not", "No"]}]''').substitute(typ=typ, first=first, second=second)


def create_identifier(verb_base, negtype, item_type):
    verb_neg_follow = verb_base + '_' + negtype + '_' + item_type
    identifier = '"{}"'.format(verb_neg_follow)

    if item_type.split('_')[0] == 'test':
        i = 1 + 4*verbs.keys().index(verb_base) 
        i += control_frames.keys().index(negtype)
    else:
        i = 1 + 4*filler_tuples.index((verb_base, item_type))

    return i, identifier


def build_item(index, identifier, verb_base, pred_tuple, frame, follow_up_frame):

    if negtype == 'negative':
        first_sent = frame(verb_base, pred_tuple)
    else:
        verb_past = verbs[verb_base]
        first_sent = frame(verb_past, pred_tuple)

    second_sent = follow_up_frame(pred_tuple)

    label = '[{}, {}]'.format(identifier, str(index))

    return item_temp(label, first_sent, second_sent)


verb_neg_list = []

test_labels = []
filler_labels = []

for verb_base in verbs.keys():
    for negtype, frame in control_frames.iteritems():

        index_control, identifier_control = create_identifier(verb_base, negtype, 'test_control')
        # index_hadto, identifier_hadto = create_identifier(verb_base, negtype, 'test_hadto')

        test_labels.append(identifier_control)
        # test_labels.append(identifier_hadto)

        for predicate_name, pred_tuple in predicates.iteritems():

            item = build_item(index_control, identifier_control, verb_base, pred_tuple, frame, follow_up_test)
            all_items.append(item)

            # if verb_base in hadto_verbs:
            #     item = build_item(index_hadto, identifier_hadto, verb_base, pred_tuple, hadto_frames[negtype], follow_up_test)
            #     all_items.append(item)
            # else:
            #     all_items.append(item)
                
    if verb_base in frames_fillers:
        for ftype, fillerframeset in frames_fillers[verb_base].iteritems():
            j = 0
            for negtype, frame in fillerframeset.iteritems():

                index, identifier = create_identifier(verb_base, negtype, ftype)
                filler_labels.append(identifier)

                if ftype == 'about':
                    follow_up_frame = follow_up_test

                else:
                    follow_up_frame = follow_up_filler
                    
                for predicate_name, pred_tuple in predicates.iteritems():

                    item = build_item(index+j, identifier, verb_base, pred_tuple, frame, follow_up_frame)
                    all_items.append(item)

                j += 1



def create_debrief_controller(size=15):
    alpha = [str(i) for i in range(10)] + ['A', 'B', 'C', 'D', 'E', 'F']
    code = ''.join(random.sample(alpha, size))
    return '[["debrief", 0], "Message", {html: "<p><center>Please enter the following code into Mechanical Turk.</center></p><p><center><b>'+code+'</b></center></p>", transfer: "click"}]'

debrief_controllers = [create_debrief_controller() for i in range(300)]

all_items += debrief_controllers


practice_items = '''
        ["practice", "Question",  {q: "Gary loves that Mary is happy.<br/><br/>Is Mary happy?", randomOrder: false, as: ["Yes", "Maybe or maybe not", "No"]}],
        ["practice", "Question",  {q: "Dave told James to go to the store.<br/><br/>Did James go to the store?", randomOrder: false, as: ["Yes", "Maybe or maybe not", "No"]}],
        ["practice", "Question",  {q: "It doesn't surprise Darlene that Jason isn't a virtuoso.<br/><br/>Is Jason a virtuoso?", randomOrder: false, as: ["Yes", "Maybe or maybe not", "No"]}],
        ["practice", "Question",  {q: "Fred was willing to pay his own way.<br/><br/>Did Fred pay his own way?", randomOrder: false, as: ["Yes", "Maybe or maybe not", "No"]}],
'''

experiment = Template('''
var shuffleSequence = seq("consent", "setcounter", "intro", "practice", "begin", rshuffle(shuffle($test_labels), shuffle($filler_labels)), "sr", "debrief");
var practiceItemTypes = ["practice"];
var manualSendResults = true;

var defaults = [
    "Message", { 
        hideProgressBar: true, 
        transfer: "keypress" 
    },
    "Form", { hideProgressBar: true },
    "Question", { hideProgressBar: true }

];

var items = [
	["consent", "Form", {
        html: { include: "consent.html" },
		validators: {age: function (s) { if (s.match(/^\d+$$/)) return true;
							else return "Bad value for age"; }}
    } ],

	["intro", "Message", {html: { include: "introduction.html" }, transfer: "click"}],

        ["setcounter", "__SetCounter__", { }],
        ["sr", "__SendResults__", { }],

        $practice_items

	["begin", "Message", {
			      html: { include: "begin.html" },
                              transfer: "click"
				} ],

    ["sep", "Separator", { }],

    $items


];
''').substitute(practice_items=practice_items, test_labels=','.join(test_labels), filler_labels=','.join(filler_labels), items=',\n'.join(all_items))

## write experiment file

with open('experiment.js', 'w') as exp_script:
    exp_script.write(experiment)
