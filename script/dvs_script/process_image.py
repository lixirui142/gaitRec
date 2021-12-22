import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def fill_img(in_path, out_path, pre_img):
    img = cv2.imread(in_path)
    #img = cv2.fastNlMeansDenoising(img,h = 10, templateWindowSize = 7)
    #img = cv2.GaussianBlur(img, (3,3),0)
    img_r = cv2.erode(img, (2,2))
    img_r = cv2.dilate(img_r, (2,2))
    # img_r = cv2.GaussianBlur(img_r, (3,3),0)
    gray=cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
    # canny=cv2.Canny(gray,40,140)
    #img_r = cv2.Laplacian(gray, -1)
    img_r = gray
    #img_r = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img=cv2.Canny(gray,60,100)
    # # if pre_img is not None:
    # #     img[mat] += pre_img[mat] // 2
    img_r = cv2.copyMakeBorder(img_r, 80, 0, 0, 0, cv2.BORDER_REPLICATE)
    cv2.imwrite(out_path, img_r)
    return img



#dvs_img = "dvs/cb_90new"
dvs_img = "G:\\program\\github\\gaitRec\\dvs\\pic\\1\\1\\1_1"
#out_dir = "dvs/cbnewblur"
out_dir = "G:\\program\\github\\gaitRec\\dvs\\pic_pro\\1\\1\\1_1" 
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


