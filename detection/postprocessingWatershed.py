import cv2
import os
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import numpy as np
from scipy.ndimage import label
import random as rng
ViabilityIndex = []
Filetag = []
selem = disk(8)
from os.path import isfile, join
from os import listdir


def RedContoursWatershed(path = ''):
    result_path = path + '/detection_output/results/'
    save_path = path + '/detection/'
    test_path = path + '/detection_output/'
    imgID = '0'


    images_num = [f for f in listdir(result_path) if isfile(join(result_path, f))]
    total = len(images_num)
    num = 0
    detection_npv = [[] for i in range(total + 20)]

    for folder in os.listdir(result_path):
        lastimgID = imgID
        mask = cv2.imread(result_path + folder, cv2.IMREAD_GRAYSCALE)
        imgID = folder.split('_')[0]
        maskID = folder.split('_')[1][0:-4]
        print(imgID)

        if imgID != lastimgID:
            RedContours1 = cv2.imread(save_path + '/' + imgID + '/images/' + imgID + '.png', cv2.IMREAD_COLOR)
            RedContours2 = cv2.imread(test_path + 'maskMap/' + imgID + '.png', cv2.IMREAD_COLOR)

        # delete black images
        counter_1 = 0
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] > 100:
                    counter_1 = counter_1 + 1
        print(counter_1)
        if counter_1 > 150:

            # logic-and SHG image
            img = cv2.imread(save_path + '/' + imgID + '/images/' + imgID + '.png', cv2.IMREAD_GRAYSCALE)
            img_masked = cv2.bitwise_and(img, img, mask=mask)  # 8bit

            # closing
            #         kernel = np.ones((5, 5), np.uint8)
            #         border = cv2.erode(mask, kernel, iterations=3)
            #         border = cv2.dilate(border, kernel, iterations=3)

            # watershed detection
            img_BGR = cv2.cvtColor(img_masked, cv2.COLOR_GRAY2BGR)
            pre_gray = cv2.GaussianBlur(img_masked, (5, 5), 0)
            img_op = opening(pre_gray, selem)

            # laplacian filtering
            kernel_1 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
            kernel_2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)  # need further optimization
            imgLaplacian = cv2.filter2D(img_op, cv2.CV_32F, kernel_2)
            sharp = np.float32(img_op)
            imgResult = sharp - imgLaplacian
            imgResult = np.clip(imgResult, 0, 255)
            imgResult = imgResult.astype('uint8')
            imgLaplacian = np.clip(imgLaplacian, 0, 255)
            imgLaplacian = np.uint8(imgLaplacian)

            _, bw = cv2.threshold(imgResult, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Perform the distance transform algorithm
            dt = cv2.distanceTransform(bw, 2, 3)
            dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
            _, dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)

            # Dilate a bit the dist image
            kernel1 = np.ones((3, 3), dtype=np.uint8)
            dt = cv2.dilate(dt, kernel1)

            # Create the CV_8U version of the distance image
            # It is needed for findContours()
            dt_8u = dt.astype('uint8')

            # Find total markers
            _, contours, _ = cv2.findContours(dt_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create the marker image for the watershed algorithm
            markers = np.zeros(dt.shape, dtype=np.int32)

            # Draw the foreground markers
            for i in range(len(contours)):
                cv2.drawContours(markers, contours, i, (i + 1), -1)

            # Draw the background marker
            cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
            # cv2.circle(img,center,radius,color,thickness,lineType,shift)
            # Perform the watershed algorithm

            imgResult = cv2.cvtColor(imgResult, cv2.COLOR_GRAY2BGR)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(mask, markers)  # Input 8-bit 3-channel image!

            # mark = np.zeros(markers.shape, dtype=np.uint8)
            mark = markers.astype('uint8')
            mark = cv2.bitwise_not(mark)

            class Point(object):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def getX(self):
                    return self.x

                def getY(self):
                    return self.y

            connects = [Point(-1, 0), Point(1, 0), Point(0, -1), Point(0, 1)]
            # delete too large images
            counter_2 = 0
            for y in range(mark.shape[0]):
                for x in range(mark.shape[1]):
                    if mark[y, x] > 100:
                        counter_2 = counter_2 + 1

            if not os.path.exists(path + '/detection_output/UW/'):
                os.mkdir(path + '/detection_output/UW/')
            if not os.path.exists(path + '/detection_output/UW/results/'):
                os.mkdir(path + '/detection_output/UW/results/')
            if not os.path.exists(path + '/detection_output/UW/RedContours1/'):
                os.mkdir(path + '/detection_output/UW/RedContours1/')
            if not os.path.exists(path + '/detection_output/UW/RedContours2/'):
                os.mkdir(path + '/detection_output/UW/RedContours2/')

            if counter_2 < 10000:

                for n in range(len(contours)):
                    x, y, w, h = cv2.boundingRect(contours[n])
                    seed = np.zeros((mark.shape[0], mark.shape[1]), dtype=np.uint8)
                    flag = np.zeros([mark.shape[0], mark.shape[1]])
                    img_result = np.zeros(([mark.shape[0], mark.shape[1]]), np.uint8)
                    seed_list = []
                    cv2.drawContours(seed, contours, n, 255, thickness=-1)
                    # cv2.imwrite(test_path + 'mask2/' + folder[0:-4] + '_seed' + str(n) + '.png', seed)
                    for y in range(mark.shape[0]):
                        for x in range(mark.shape[1]):
                            if seed[y, x] > 100:
                                seed_list.append(Point(y, x))

                    class_k = 1

                    while len(seed_list) > 0:
                        seed_tmp = seed_list[0]
                        seed_list.pop(0)
                        flag[seed_tmp.x, seed_tmp.y] = class_k
                        img_result[seed_tmp.x, seed_tmp.y] = 255
                        for i in range(4):
                            tmpX = seed_tmp.x + connects[i].x
                            tmpY = seed_tmp.y + connects[i].y

                            if (tmpX < 0 or tmpY < 0 or tmpX >= 256 or tmpY >= 256):
                                continue

                            if ((mark[tmpX, tmpY] > 100) and (flag[tmpX, tmpY] == 0)):
                                img_result[tmpX, tmpY] = 255
                                flag[tmpX, tmpY] = class_k
                                seed_list.append(Point(tmpX, tmpY))
                            if mark[tmpX, tmpY] < 100:

                                RedContours1[tmpX, tmpY] = (0, 0, 255)
                                RedContours1[seed_tmp.x, seed_tmp.y] = (0, 0, 255)
                                RedContours2[tmpX, tmpY] = (0, 0, 255)
                                RedContours2[seed_tmp.x, seed_tmp.y] = (0, 0, 255)
                                detection_npv[num].append([tmpX, tmpY])

                    # save masks into directory
                    UW_path = test_path + 'UW/'
                    if not os.path.exists(save_path + imgID + '/masks2/'):
                        os.mkdir(save_path + imgID + '/masks2/')

                    # cv2.imwrite(test_path + 'mask2/' + folder[0:-4] + '_' + str(n+1) + '.png', img_result)
                    cv2.imwrite(UW_path + 'results/' + folder[0:-4] + '_' + str(n + 1) + '.png', img_result)
                    cv2.imwrite(save_path + imgID + '/masks2/' + folder[0:-4] + '_' + str(n+1) + '.png', img_result)
                    # save Red contours images
                    num = num + 1

                    cv2.imwrite(UW_path + 'RedContours1/' + imgID + '.png', RedContours1)
                    cv2.imwrite(UW_path + 'RedContours2/' + imgID + '.png', RedContours2)

        # num = num + 1
    np.save('classification/npv/COUNTING/UW/detection.npy', detection_npv)
    np.save('classification/npv/VIABILITY/UW/detection.npy', detection_npv)