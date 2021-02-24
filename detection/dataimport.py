from os import listdir
from os.path import isfile, join
import os
import cv2
import numpy as np

def dataImport(in_path = ''):
    save_path = in_path + '/detection/'

    onlyfiles = [f for f in listdir(in_path) if isfile(join(in_path, f))]
    for folder in onlyfiles:
        ind = folder. split('RGB')[0]
        print(ind)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(save_path + ind):
            os.mkdir(save_path + ind)
        if not os.path.exists(save_path + ind + '/masks/'):
            os.mkdir(save_path + ind + '/masks/')
        if not os.path.exists(save_path + ind + '/images/'):
            os.mkdir(save_path + ind + '/images/')
        raw = cv2.imread(in_path + '/' + folder)
        b, g, r = cv2.split(raw)
        img = cv2.bitwise_not(b)

        equ = cv2.equalizeHist(img)

        new_image = np.zeros(img.shape, img.dtype)
        alpha = 2.5  # Simple contrast control
        beta = -350  # Simple brightness control
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                new_image[y, x] = np.clip(alpha * equ[y, x] + beta, 0, 255)

        # new_image = cv2.medianBlur(new_image, 13)

        new_imageBGR = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(save_path + ind +'/images/' + ind + '.png', new_imageBGR)