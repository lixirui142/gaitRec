f1 = "script/process_joints.py"
f2 = "script/process_joints0.py"

with open(f1, 'r') as p1, open(f2, 'r') as p2:
    for l1, l2 in zip(p1.readlines(), p2.readlines()):
        if l1 != l2:
            print(l1,l2)