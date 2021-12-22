import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def find_region(condition):
    max_length = 0 
    return_s = 0
    return_e = 0
    for start, stop in contiguous_regions(condition):
        t = stop - start
        if max_length < t:
            return_s = start
            return_e = stop
            max_length = t
    return return_s, return_e

def crop_and_save(img_path, out_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 1)
    gray = cv2.fastNlMeansDenoising(gray,h = 10, templateWindowSize = 7)

    sm = gray.sum(0)
    condition = sm > min(sm)
    s,e = find_region(condition)
    l = e - s
    s = max(0, s - l // 4)
    e = min(gray.shape[1] - 1, e + l // 4)
    img = img[:, s:e]

    sm = gray.sum(1)
    condition = sm > min(sm)
    s,e = find_region(condition)
    l = e - s
    s = max(0, s - l // 4)
    e = min(gray.shape[0] - 1, e + l // 4)
    img = img[s:e, :]

    cv2.imwrite(out_path, img)



dvs_img = "dvs/pic"
out_dir = "dvs/crop_img"
for root, dirs, files in os.walk(dvs_img):
    out_root = root.replace(dvs_img, out_dir)
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    for file in files:
        if ".png" in file:
            in_path = os.path.join(root, file)
            out_path = os.path.join(out_root, file)
            crop_and_save(in_path, out_path)


