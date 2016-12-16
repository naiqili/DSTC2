from scripts.dataset_walker import *
import itertools

dataroot_path = "data"
dataset_train = dataset_walker("dstc2_train", dataroot=dataroot_path, \
                                                              labels=True)
dataset_dev = dataset_walker("dstc2_dev", dataroot=dataroot_path, \
                                                            labels=True)

sys_acts = []
user_acts = []
tasks = []

def dialog_acts_to_str(acts):
    res = []
    for act in acts:
        s = act["act"] + "("
        if len(act["slots"]) > 0:
            if act["act"] == "request":
                s = s + act["slots"][0][1]
            else:
                s = s + act["slots"][0][0]
        s = s + ")"
        res.append(s)
    return res

cnt = 0
for call in itertools.chain(dataset_train, dataset_dev) :
    cnt = cnt + 1
    for _turn, _label in call :
        user_acts = user_acts + dialog_acts_to_str(_label["semantics"]["json"])
        sys_acts  = sys_acts + dialog_acts_to_str(_turn["output"]["dialog-acts"])
    tasks.append(call.task["goal"])

uni_user_acts = dict([(act, user_acts.count(act)) for act in set(user_acts)])
uni_sys_acts = dict([(act, sys_acts.count(act)) for act in set(sys_acts)])

with open('./output/user_acts.txt', 'w') as f_user, \
     open('./output/sys_acts.txt', 'w') as f_sys, \
     open('./output/tasks.txt', 'w') as f_tasks:
    f_user.write("Num of unique user actions: %d\n" % len(uni_user_acts))
    f_user.write("Counts:\n")
    for (act, cnt) in uni_user_acts.items():
        f_user.write("%s: %d\n" % (act, cnt))
    
    f_sys.write("Num of unique sys actions: %d\n" % len(uni_sys_acts))
    f_sys.write("Counts:\n")
    for (act, cnt) in uni_sys_acts.items():
        f_sys.write("%s: %d\n" % (act, cnt))

    for task in tasks:
        fields = ["request-slots", "constraints", "text"]
        for field in fields:
            f_tasks.write(field + ": ")
            f_tasks.write(str(task[field]) + "\n")
        f_tasks.write("\n")
