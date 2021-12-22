import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def extract_joints(in_path, image_out_path, json_out_path):
    cmd = "cd G:\program\github\gaitRec\openpose && \
        bin\OpenPoseDemo.exe --image_dir {} --write_images {} --write_json {} --display 0".format(in_path, image_out_path, json_out_path)
    os.system(cmd)
    

full_img_dir = "G:\program\github\gaitRec\dvs\cbblur_n"
img_dir = "dvs\cbblur_n"
json_out_dir = "data\joints_json"
image_out_dir = "data\joints_img"

for root, dirs, files in os.walk(full_img_dir):
    image_out_root = root.replace(img_dir, image_out_dir)
    json_out_root = root.replace(img_dir, json_out_dir)
    if not os.path.exists(image_out_root):
        os.mkdir(image_out_root)
    if not os.path.exists(json_out_root):
        os.mkdir(json_out_root)
    if len(files) != 0 and ".png" in files[0]:
        extract_joints(root, image_out_root, json_out_root)
    # flag = 0
    # for file in files:
    #     if ".png" in file:
    #         flag = 1
    #         in_path = os.path.join(root, file)
    #         image_out_path = os.path.join(image_out_root, file)
    #         json_out_path = os.path.join(json_out_root, file)
    #         extract_joints(in_path, image_out_path, json_out_path)
    # if flag == 1:
    #     break


