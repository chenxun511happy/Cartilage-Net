import cv2
from tkinter import messagebox
from os import listdir
from os.path import isfile, join
import os

def cropPositionMap(i = 0):
    x, y, w, h = 0, 0, 256, 256
    if i == 0 or i == 1 or i == 2 or i == 3:
        x, y, w, h = 0, 256*i, 256, 256
    elif i == 4 or i == 5 or i == 6 or i == 7:
        x, y, w, h = 256, 256*(i-4), 256, 256
    elif i == 8 or i == 9 or i == 10 or i == 11:
        x, y, w, h = 512, 256 * (i - 8), 256, 256
    elif i == 12 or i == 13 or i == 14 or i == 15:
        x, y, w, h = 768, 256 * (i - 12), 256, 256
    return x, y, w, h

def cropImg(analysis_type = dict(), in_path = ''):

    if not int(analysis_type["pixel_size"][0]) in [256, 512, 1024]:
       messagebox.showerror("error", "Input pixel size = 256, 512, 1024")
       raise ValueError("Input pixel size = 256, 512, 1024")
    if not int(analysis_type["pixel_size"][0]) == 256:
        save_path = in_path + '/precrop/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    path_type = []
    onlyfiles = [f for f in listdir(in_path) if isfile(join(in_path, f))]
    for folder in onlyfiles:
        raw = cv2.imread(in_path + '/' + folder)
        if raw.shape[0] in [256, 512, 1024] and raw.shape[1] in [256, 512, 1024]:
            if raw.shape[0] == 256:
                path_type.append(256)
                # cv2.imwrite(save_path + folder, raw)
            elif raw.shape[0] == 512:
                path_type.append(512)
                for i in range(4):
                    if i == 0:
                        x, y, w, h = 0, 0, 256, 256
                    elif i == 1:
                        x, y, w, h = 0, 256, 256, 256
                    elif i == 2:
                        x, y, w, h = 256, 0, 256, 256
                    else:
                        x, y, w, h = 256, 256, 256, 256
                    newraw = raw[y:y + h, x:x + w]
                    cv2.imwrite(save_path + folder.split('R')[0] + str(i) + 'RGB' + folder.split('B')[1], newraw)
            elif raw.shape[0] == 1024:
                path_type.append(1024)
                for i in range(16):
                    x, y, w, h = cropPositionMap(i)
                    newraw = raw[y:y + h, x:x + w]
                    cv2.imwrite(save_path + folder.split('R')[0] + str(format(i, '02d')) + 'RGB' + folder.split('B')[1], newraw)
        else:
            raise ValueError("Input pixel size = 256, 512, 1024")


    print("crop complete")
    if 256 in path_type:
        return in_path
    else:
        return save_path