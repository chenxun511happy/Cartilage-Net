import cv2
import os
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import numpy as np
from os.path import isfile, join
from os import listdir


def RedContoursUnet(path = ''):
    result_path = path + '/detection_output/results/'
    save_path = path + '/detection/'
    test_path = path + '/detection_output/'

    class Point(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def getX(self):
            return self.x

        def getY(self):
            return self.y


    connects = [Point(-1, 0), Point(1, 0), Point(0, -1), Point(0, 1)]
    imgID = '0'

    images_num = [f for f in listdir(result_path) if isfile(join(result_path, f))]
    total = len(images_num)
    num = 0
    detection_npv = [[] for i in range(total)]
    for folder in os.listdir(result_path):
        lastimgID = imgID
        mask = cv2.imread(result_path + folder, cv2.IMREAD_GRAYSCALE)
        imgID = folder.split('_')[0]
        # maskID = folder.split('_')[1][0:-4]
        print(imgID)

        if imgID != lastimgID:
            RedContours1 = cv2.imread(save_path + '/' + imgID + '/images/' + imgID + '.png', cv2.IMREAD_COLOR)
            RedContours2 = cv2.imread(test_path + 'maskMap/' + imgID + '.png', cv2.IMREAD_COLOR)

        # delete black images
        counter = 0
        class_k = 1
        seed_list =[]
        flag = np.zeros([mask.shape[0], mask.shape[1]])
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] > 100:
                    counter = counter + 1
                    seed_list.append(Point(y, x))

        if not os.path.exists(path + '/detection_output/RedContours1/'):
            os.mkdir(path + '/detection_output/RedContours1/')
        if not os.path.exists(path + '/detection_output/RedContours2/'):
            os.mkdir(path + '/detection_output/RedContours2/')

        if counter > 150:

            cv2.imwrite(save_path + '/' + imgID + '/masks/' + folder, mask)

            while len(seed_list) > 0:
                seed_tmp = seed_list[0]
                seed_list.pop(0)
                flag[seed_tmp.x, seed_tmp.y] = class_k
                for i in range(4):
                    tmpX = seed_tmp.x + connects[i].x
                    tmpY = seed_tmp.y + connects[i].y

                    if (tmpX < 0 or tmpY < 0 or tmpX >= 256 or tmpY >= 256):
                        continue

                    if mask[tmpX, tmpY] < 100:
                        RedContours1[tmpX, tmpY] = (0, 0, 255)
                        RedContours1[seed_tmp.x, seed_tmp.y] = (0, 0, 255)
                        RedContours2[tmpX, tmpY] = (0, 0, 255)
                        RedContours2[seed_tmp.x, seed_tmp.y] = (0, 0, 255)
                        detection_npv[num].append([tmpX, tmpY])

            cv2.imwrite(test_path + 'RedContours1/' + imgID + '.png', RedContours1)
            cv2.imwrite(test_path + 'RedContours2/' + imgID + '.png', RedContours2)
        num = num + 1
    np.save('classification/npv/COUNTING/UNET/detection.npy', detection_npv)
    np.save('classification/npv/VIABILITY/UNET/detection.npy', detection_npv)