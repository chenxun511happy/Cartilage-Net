import cv2
import os
from os import listdir
from os.path import isfile, join
from PIL import Image

def splitintoClusters(path = '', MODEL_YTPE = 'UNET'):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    if MODEL_YTPE == 'UW':
        path_UnetWatershed = path + '/detection_output/'
        path_croppedUW = path + '/detection_output/UW/cropped_images/'
        path_resizedUW = path + '/detection_output/UW/resized_images/'
    elif MODEL_YTPE == 'UNET':
        path_croppedUW = path + '/detection_output/cropped_images/'
        path_resizedUW = path + '/detection_output/resized_images/'
        path_UnetWatershed = path + '/detection_output/'
    else:
        path_croppedUW = path + '/detection_output/watershed/cropped_images/'
        path_resizedUW = path + '/detection_output/watershed/resized_images/'
        path_UnetWatershed = path + '/detection_output/'

    if not os.path.exists(path_croppedUW):
        os.mkdir(path_croppedUW)
    if not os.path.exists(path_resizedUW):
        os.mkdir(path_resizedUW)
    # Extract
    for folder in onlyfiles:
        img = cv2.imread(path + '\\' + folder,cv2.IMREAD_COLOR)
        imgID = folder.split('R')[0]
        print(imgID)
        if MODEL_YTPE == 'UNET':
            path_mask = path_UnetWatershed + 'results/'
        elif MODEL_YTPE == 'UW':
            path_mask = path_UnetWatershed + 'UW/results/'
        else:
            path_mask = path_UnetWatershed + 'watershed/results/'

        for idx, submask in enumerate(os.listdir(path_mask)):
            if submask.split('_')[0] == imgID:
                if MODEL_YTPE == 'UNET':
                    clusterids = submask.split('_')[1].split('.')[0]
                else:
                    clusterids = submask[submask.find("_")+1:submask.find(".")]
                mask = cv2.imread(path_mask+ submask, cv2.IMREAD_GRAYSCALE)
                result = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite(path + 'detection_output/detectionResult/' + imgID + '_' + str(clusterids) +'.png',result)

        # Crop
                imagecon, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = sorted(cnts, key=cv2.contourArea)[-1]
                x, y, w, h = cv2.boundingRect(cnt)
                dstmask = result[y:y + h, x:x + w]
                cv2.imwrite(path_croppedUW + imgID + '_' + str(clusterids) + '.jpeg', dstmask)

        # Resize
                original_image = Image.open(path_croppedUW + imgID + '_' + str(clusterids) + '.jpeg')
                old_size = original_image.size
                new_size = (100, 100)
                new_im = Image.new("RGB", new_size)
                new_im.paste(original_image, (int((new_size[0] - old_size[0]) / 2), int((new_size[1] - old_size[1]) / 2)))
                resized_image = new_im.resize((256, 256), Image.ANTIALIAS)
                resized_image.save(path_resizedUW + imgID + '_' + str(clusterids) + '.jpeg')
