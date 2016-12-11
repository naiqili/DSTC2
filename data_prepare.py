from scripts.dataset_walker import *
import scripts.misc as misc
import json, os

# Path for input
dataroot_path = "data"
ontology_path = "./scripts/config/ontology_dstc2.json"

# Path for output
dict_path = '/tmp/dict.txt'
method_train_path = "./tmp/method.train"
method_dev_path = "./tmp/method.dev"

# Dictionary
# </s>: end of sentence, </t>: end of turn
word2int = {"</s>": 0, "</t>": 1, "<oov>": 2}
wordcnt = 3

def word2int_fun(w):
    global word2int, wordcnt
    if w in word2int:
        return word2int[w]
    word2int[w] = wordcnt
    wordcnt = wordcnt + 1
    return word2int[w]

def save_dict(path=dict_path):
    with open(path, 'w') as f:
        l = word2int.items()
        l.sort(key=lambda t: t[1])
        for (k, v) in l:
            f.write("%s %d\n" % (k, v))

# Prepare ontology

with open(ontology_path) as f_ont:
    ont = json.load(f_ont)
    ont_req = ont["requestable"] # list = [addr, area, ...]
    ont_meth = ont["method"] # list = [none, byconstraint, byname, ...]
    ont_info = ont["informable"] # dict = [food: [], pricerange: [],...]
    # The goals are <informable-value> pair

for w in ont_req:
    if not w in word2int:
        word2int[w] = wordcnt
        wordcnt = wordcnt + 1

for w in ont_meth:
    if not w in word2int:
        word2int[w] = wordcnt
        wordcnt = wordcnt + 1

for (head, list) in ont_info.items():
    for w in [head] + list:
        if not w in word2int:
            word2int[w] = wordcnt
            wordcnt = wordcnt + 1


# Prepare data
#
# train_data = [label_log_1, ..., label_log_n]
# Each label_log = [id, TURN_1, ..., TURN_i]
# TURN_i is dict:
# { system_act: parsed INT string # (dialog_act, slot)
#   asr_hyp1: (string, score) # best ASR result
#   method_label: string
#   goal_label: dict(slot, value) # slot in ont_info,
#                                 # value in ont_info[slot]
#   request_label: list(string) # string in ont_req, can be EMPTY

# dialog_acts is a list of dialog_act
# each dialog_act is a dict {act: string,
#                            slots: [slot, value]}
# During the parsing, string are replace by int
# 
def parse_dialog_acts(dialog_acts):
    res = ""
    for dialog_act in dialog_acts:
        res =  res + str(word2int_fun(dialog_act["act"])) + " "
        for slot in dialog_act["slots"]:
            # if the slot is requestable, we skip its content
            # e.g. phone XXXXXX, I believe XXXXX is not important
            if slot[0] in ont_req:
                res = res + str(word2int_fun(slot[0])) + " "
                continue
            for v in slot:
                if v == "slot": continue
                res = res + str(word2int_fun(v)) + " "
    return res

# Training data

print "Loading training data ..."

dataset_train = dataset_walker("dstc2_train", dataroot=dataroot_path, \
                               labels=True)
train_data = []

for call in dataset_train :
    label_log = []
    label_log.append(call.log["session-id"])
    for _turn, _label in call :
        turn = {}
        dialog_acts = _turn["output"]["dialog-acts"]
        act_str = parse_dialog_acts(dialog_acts)
        turn["system_act"] = act_str
        turn["asr_hyp1"] = \
            (_turn["input"]["live"]["asr-hyps"][0]["asr-hyp"],
             _turn["input"]["live"]["asr-hyps"][0]["score"])
        turn["method_label"] = _label["method-label"]
        turn["goal_label"] = _label["goal-labels"] # a dict
        turn["request_label"] = _label["requested-slots"]
        label_log.append(turn)
        for w in turn["asr_hyp1"][0].split():
            word2int_fun(w)
    train_data.append(label_log)

# Dev data

print "Loading dev data ..."

dataset_dev = dataset_walker("dstc2_dev", dataroot=dataroot_path, \
                               labels=True)
dev_data = []

for call in dataset_dev :
    label_log = []
    label_log.append(call.log["session-id"])
    for _turn, _label in call :
        turn = {}
        dialog_acts = _turn["output"]["dialog-acts"]
        act_str = parse_dialog_acts(dialog_acts)
        turn["system_act"] = act_str
        turn["asr_hyp1"] = \
            (_turn["input"]["live"]["asr-hyps"][0]["asr-hyp"],
             _turn["input"]["live"]["asr-hyps"][0]["score"])
        turn["method_label"] = _label["method-label"]
        turn["goal_label"] = _label["goal-labels"] # a dict
        turn["request_label"] = _label["requested-slots"]
        label_log.append(turn)
        for w in turn["asr_hyp1"][0].split():
            word2int_fun(w)
    dev_data.append(label_log)

# Build training\dev data for different prediction
# All training\dev data in int form
# Each log in 2 lines:
# TURN_1 \t TURN_2 ... \t TURN_n
# PRED_1 \t PRED_2 ... \t PRED_n

# Training data for method

print "Building " + method_train_path + " ..."
max_train_len = 0
with open(method_train_path, 'w') as f:
    for label_log in train_data:
        line1 = ""
        line2 = ""
        for turn in label_log[1:]: # first entry is id, skip
            hyp_s = turn["asr_hyp1"][0]
            hyp_i = ""
            for w in hyp_s.split():
                hyp_i = hyp_i + str(word2int_fun(w)) + " "
            line1 = line1 + turn["system_act"] + " " + str(word2int["</s>"]) + \
                    " " + hyp_i + str(word2int["</t>"]) + " "
            max_train_len = max(max_train_len, len(line1.split()))
            # Prediction is a single int for each turn,
            # which is the index of method ontology
            line2 = line2 + str(ont_meth.index(turn["method_label"])) + " "
        f.write("%s\n%s\n" % (line1, line2))

# Dev data for method

print "Building " + method_dev_path + " ..."
max_dev_len = 0
with open(method_dev_path, 'w') as f:
    for label_log in dev_data:
        line1 = ""
        line2 = ""
        for turn in label_log[1:]: # first entry is id, skip
            hyp_s = turn["asr_hyp1"][0]
            hyp_i = ""
            for w in hyp_s.split():
                hyp_i = hyp_i + str(word2int_fun(w)) + " "
            line1 = line1 + turn["system_act"] + " " + str(word2int["</s>"]) + \
                    " " + hyp_i + str(word2int["</t>"]) + " "
            max_dev_len = max(max_dev_len, len(line1.split()))
            # Prediction is a single int for each turn,
            # which is the index of method ontology
            line2 = line2 + str(ont_meth.index(turn["method_label"])) + " "
        f.write("%s\n%s\n" % (line1, line2))        

# Save

save_dict()

# Output summary

print "Word count:", wordcnt
print
print "Num of req_ont, method_ont, info_ont:", \
    len(ont_req), len(ont_meth), len(ont_info)
print
print "Num of train data:", len(train_data)
print
print "Num of dev data:", len(dev_data)
print
print "Max train seq len", max_train_len
print
print "Max dev seq len", max_dev_len
print
print "Example of dict:"
os.system("head " + dict_path)
print
print "Example of method.train:"
os.system("head " + method_train_path)
print
print "Example of method.dev:"
os.system("head " + method_dev_path)
