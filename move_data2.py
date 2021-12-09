
import os
import shutil
rp = "data\gait_joints_json_90"
np = "data\gait_prime_joints"
type = ["nm", "cl", "bg"]
for root, dirs, files in os.walk(rp):
    if len(files) != 0: #G:\program\github\gaitRec\data\gait_joints_json_90\1\1\1_1
        hid, tid, sid = root.split('\\')[-3:]        
        nname = "{:0>3d}-{}-{:0>2d}-090".format(int(hid), type[int(tid) - 1], int(sid.split('_')[-1]))
        ndir = os.path.join(np, nname)

        if not os.path.exists(ndir):
            os.makedirs(ndir)
        for file in files:
            nfilename = "{}_{:0>12d}_keypoints.json".format(nname, int(file.split('_')[0]))
            npath = os.path.join(ndir, nfilename)
            shutil.move(os.path.join(root, file), npath)