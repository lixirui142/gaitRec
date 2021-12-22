import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def fill_img(in_path, out_path, pre_img):
    img = cv2.imread(in_path)
    #img = cv2.fastNlMeansDenoising(img,h = 10, templateWindowSize = 7)
    img_r = cv2.GaussianBlur(img, (5,5),0)
    #img_r = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img=cv2.Canny(gray,60,100)
    # # if pre_img is not None:
    # #     img[mat] += pre_img[mat] // 2
    #img_r = cv2.copyMakeBorder(img_r, 40, 40, 0, 0, cv2.BORDER_REPLICATE)
    cv2.imwrite(out_path, img_r)
    return img



#dvs_img = "dvs/cb_90new"
dvs_img = "G:/program/github/gaitRec/dvs/cb_0_denoise/2/1"
#out_dir = "dvs/cbnewblur"
out_dir = "G:/program/github/gaitRec/dvs/cb_0_blur/2/1" 
for root, dirs, files in os.walk(dvs_img):
    out_root = root.replace(dvs_img, out_dir)
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    files.sort(key=lambda x: int(x.strip(".png")))
    pre_img = None
    flag = 0
    for file in files:
        if ".png" in file:
            flag = 1
            in_path = os.path.join(root, file)
            out_path = os.path.join(out_root, file)
            pre_img = fill_img(in_path, out_path, pre_img)
            
    if flag == 1:
        break


