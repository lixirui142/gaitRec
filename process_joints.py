import os
import json
from matplotlib.pyplot import bone
import torch

data_dir = "G:\program\github\gaitRec\data\gait_prime_joints"
ndir = "G:\program\github\gaitRec\data\gait_prime_joints_process"

for root, dirs, files in os.walk(data_dir):
    if len(files) > 0:
        files.sort(key=lambda x: x.split('_')[1])
        joints = []
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                str = f.read()
            js_data = json.loads(str)
            people = js_data['people']
            if len(people) > 0 and 'pose_keypoints_2d' in people[0]:
                joints.append(people[0]['pose_keypoints_2d'])
            else:
                joints.append([0.0 for i in range(18 * 3)])
        joints = torch.tensor(joints)
        joints = joints.view(len(joints), -1, 3)

        joints = joints[:,:14,:].contiguous()
        bone_len1 = (joints[:,1,:2] - joints[:,8,:2]).norm(dim = -1)
        bone_len2 = (joints[:,1,:2] - joints[:,11,:2]).norm(dim = -1)
        bone_weight1 = (joints[:, 1, 2] * joints[:, 8, 2]).sqrt()
        bone_weight2 = (joints[:, 1, 2] * joints[:, 11, 2]).sqrt()
        bone_weight_norm = bone_weight1.sum() + bone_weight2.sum() + 1e-10
        bone_weight1 /= bone_weight_norm 
        bone_weight2 /= bone_weight_norm

        norm_len = (bone_len1 * bone_weight1).sum() + (bone_len2 * bone_weight2).sum()
        joints[:,:,:2] = (joints[:,:,:2] - joints[:,1:2,:2]) / norm_len

        l = len(joints)
        joints = joints.view(-1, 3)
        joints[joints[:, -1] == 0] = 0
        joints = joints.view(l, -1, 3)
        
        jp = torch.cat([joints[1:], joints[-2:-1]], dim = 0)
        jn = torch.cat([joints[1:2], joints[:-1]], dim = 0)

        for i in range(joints.shape[1]):
            idx = joints[:, i, -1] == 0
            joints[idx, i, :2] = (jp[idx, i, :2] + jn[idx, i, :2]) / 2

        joints = joints.view(l, -1)

        for i, file in enumerate(files):
            with open(os.path.join(root, file), 'r') as f:
                str = f.read()
            js_data = json.loads(str)
            ndata = [{"pose_keypoints_2d": joints[i].tolist()}]
            js_data['people'] = ndata
            nroot = root.replace(data_dir, ndir)
            if not os.path.exists(nroot):
                os.makedirs(nroot)
            with open(os.path.join(nroot, file), 'w') as f:
                json.dump(js_data, f)