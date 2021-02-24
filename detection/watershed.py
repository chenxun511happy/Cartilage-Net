import cv2
import os
from os import listdir
from os.path import isfile, join
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
# Threshold make mask
selem = disk(8)
import numpy as np
import csv
from PIL import Image, ImageOps
import numpy
from scipy.ndimage import label
ViabilityIndex = []
Filetag = []

def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None,iterations=1)  #1 or 5

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl

def waterSHED(folder_test = ''):
    onlyfiles = [f for f in listdir(folder_test) if isfile(join(folder_test, f))]


    images_num = [f for f in listdir(folder_test) if isfile(join(folder_test, f))]
    total = len(images_num)
    num = 0
    detection_npv = [[] for i in range(total*30)]



    for index, folder in enumerate(onlyfiles):


        pre_img = cv2.imread(folder_test + '\\' + folder)

        pre_img_blue = pre_img[:, :, 0]
        pre_img_invert = cv2.bitwise_not(pre_img_blue)

        new_image = np.zeros(pre_img_invert.shape, pre_img_invert.dtype)

        alpha = 2.5  # Simple contrast control
        beta = -350  # Simple brightness control

        for y in range(pre_img_invert.shape[0]):
            for x in range(pre_img_invert.shape[1]):
                new_image[y, x] = np.clip(alpha * pre_img_invert[y, x] + beta, 0, 255)

        img_gray = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
        if index == 0:
            RedContours1 = img_gray
            RedContours2 = img_gray
        # pre_img = cv2.imread('SHG.tif')
        #     pre_gray = cv2.cvtColor(pre_img_blue, cv2.COLOR_BGR2GRAY)
        #     pre_gray = cv2.GaussianBlur(pre_img_invert, (5, 5), 0)

        pre_gray = cv2.GaussianBlur(new_image, (5, 5), 0)
        img_op = opening(pre_gray, selem)
        _, img_bin = cv2.threshold(img_op, 0, 255,
                                   cv2.THRESH_OTSU)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                                   numpy.ones((3, 3), dtype=int))

        result = segment_on_dt(img_gray, img_bin)
        result_sub = segment_on_dt(img_gray, img_bin)
        mark = result.astype('uint8')
        # mark = cv2.bitwise_not(mark)

        # find contours
        imagecon, cnts, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnts, key=cv2.contourArea)[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        # if w > 100 or h > 100:
        #     raise ('crop image too large')
        print(len(cnts))
        # save red contours
        if not os.path.exists(folder_test + '/detection_output/watershed/'):
            os.mkdir(folder_test + '/detection_output/watershed/')
        if not os.path.exists(folder_test + '/detection_output/watershed/RedContours1/'):
            os.mkdir(folder_test + '/detection_output/watershed/RedContours1/')
        if not os.path.exists(folder_test + '/detection_output/watershed/RedContours2/'):
            os.mkdir(folder_test + '/detection_output/watershed/RedContours2/')
        if not os.path.exists(folder_test + '/detection_output/watershed/results/'):
            os.mkdir(folder_test + '/detection_output/watershed/results/')

        if True:
            result[result > 1] = 255
            result_sub[result_sub != 255] = 0
            result_sub = cv2.dilate(result_sub, None)
            position_circle = (result_sub == 255)
            # save npv
            for i in range(result_sub.shape[0]):
                for j in range(result_sub.shape[1]):
                    if i == 0 or i == 1 or i == 255 or i == 254 or j == 0 or j == 1 or j == 255 or j == 254:
                        position_circle[i, j] = False

            img_gray[position_circle] = (0, 0, 200)
            cv2.imwrite(folder_test + '/detection_output/watershed/RedContours1/' + folder[0:8] + '.tif', img_gray)

            kernel = np.ones((1, 1), np.uint8)
            result_sub[result_sub != 255] = 0
            result_sub = cv2.dilate(result_sub, kernel, iterations=1)
            color_img = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            color_main = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            # save npv
            for i in range(result_sub.shape[0]):
                for j in range(result_sub.shape[1]):
                    if i == 0 or i == 1 or i == 255 or i == 254 or j == 0 or j == 1 or j == 255 or j == 254:
                        position_circle[i, j] = False
            color_img[position_circle] = (0, 0, 200)
            cv2.imwrite(folder_test + '/detection_output/watershed/RedContours2/' + folder[0:8] + '.tif', color_img)

        # save cluster images
        for i in range(len(cnts)):
            seed = np.zeros((mark.shape[0], mark.shape[1]), dtype=np.uint8)
            img_result = np.zeros((mark.shape[0], mark.shape[1]), dtype=np.uint8)
            cv2.drawContours(seed, cnts, i, 255, thickness = 1)
            cv2.drawContours(img_result, cnts, i, 255, thickness=-1)
            # cv2.imwrite(folder_test + '/detection_output/watershed/RedContours2/' + folder[0:-4] + '_seed' + str(i) + '.png', seed)
            cv2.imwrite(folder_test + '/detection_output/watershed/results/' + folder[0:5] + '_' + str(i) + '.png',
                img_result)

            for y in range(mark.shape[0]):
                for x in range(mark.shape[1]):
                    if seed[y, x] > 100:
                        detection_npv[num].append([y, x])

            num = num + 1

    np.save('classification/npv/COUNTING/WATERSHED/detection.npy', detection_npv)
    np.save('classification/npv/VIABILITY/WATERSHED/detection.npy', detection_npv)