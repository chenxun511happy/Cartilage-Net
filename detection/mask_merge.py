import cv2
import os
import numpy as np



def maskMerge(path = ''):

    path_test = path + '/detection/'
    id_img = next(os.walk(path_test))[1]
    for i in range(len(id_img)):
        path_img = path_test + str(id_img[i]) + '/images/'
        path_masks = path_test + str(id_img[i]) + '/masks/'
        print(str(id_img[i]))
        img = cv2.imread(path_img + str(id_img[i]) + '.png', cv2.IMREAD_GRAYSCALE)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for folder in os.listdir(path_masks):
            sub_mask = cv2.imread(path_masks + folder, cv2.IMREAD_GRAYSCALE)
            mask = cv2.bitwise_or(mask, sub_mask)

        if not os.path.exists(path + '/detection_output/maskMap/'):
            os.mkdir(path + '/detection_output/maskMap/')
        cv2.imwrite(path + '/detection_output/' + 'maskMap/' + str(id_img[i]) + '.png', mask)

