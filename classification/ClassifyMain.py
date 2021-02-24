from classification.image_to_array import imageTonpv
from classification.cnn_class import cnn_class
import csv
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

def finalReport(label_ids = [], cids_test = [], class_count=[], class_viability =[], path = '', model='' ):
    viability_ratio = []
    num_counting = []
    num_viability = []
    label_set = sorted(set(label_ids))
    for img_ids in label_set:
        count = 0
        viability = 0
        for index, ids in enumerate(label_ids):
          if ids == img_ids:
              if class_count[index] < class_viability[index]:
                  class_viability[index] = class_count[index]
              count = count + class_count[index]
              viability = viability + class_viability[index]
        if count < viability:
             viability = count
        # fix bug
        if count == 0:
            viability_ratio.append(0)
        else:
            viability_ratio.append(float(viability/count))
        num_counting.append(count)
        num_viability.append(viability)

    label_format = []
    for index, ids in enumerate(label_set):
        label_format.append(str(format(ids, '05d')) + '_')

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    out = open(path + '/classification/' + str(timestr) + 'FINAL_REPORT' + model + '_' + 'CNN_csv.csv', 'a',
                   newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(label_format)
    csv_write.writerow(viability_ratio)
    csv_write.writerow(num_counting)
    csv_write.writerow(num_viability)

def saveasSpectrum(label_ids = [], cids_test = [], class_count=[], class_viability =[],path = '', model='' ):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    save_path = path + '/classification/'
    DETECTION_test = np.load('classification/npv/VIABILITY/' + model + '/detection.npy')
    label_set = sorted(set(label_ids))

    for index, folder in enumerate(onlyfiles):
        raw = cv2.imread(path + '\\' + folder)
        ind = folder.split('RGB')[0]
        new_imageBGR = raw
        for index_ids, ids in enumerate(label_ids):
                if list(label_set)[index] == ids and not len(DETECTION_test[index_ids]) == 0:
                    count = class_count[index_ids]
                    viability = class_viability[index_ids]
                    if count < viability:
                        viability = count
                    if count == 0:
                        green_level = 0
                        red_level = 255
                        blue_level = 255
                    else:
                        green_level = int(255 * viability / count)
                        red_level = int(255 * (1 - viability / count))
                        blue_level = 0
                    color_spectrum = (blue_level, green_level, red_level)

                    for position in DETECTION_test[index_ids]:
                        new_imageBGR[position[0], position[1]] = color_spectrum

        cv2.imwrite(save_path + ind + model + '.png', new_imageBGR)



def classifyMain(folder_test = '', folder_train = '', analysis_type = dict()):

    # counting model and viability model
    if not analysis_type["predict_type"] == 0:
        if analysis_type["model_type"] == 1:
            # step 1.0 background preprocess and npv convert
            imageTonpv(folder_test, 'UNET')

            # step 2.0 load model weight and predict
            label_ids, cids_test, class_count = cnn_class(folder_test, 'UNET', 'COUNTING')
            label_ids, cids_test, class_viability = cnn_class(folder_test, 'UNET', 'VIABILITY')

            # step 3.0 save final csv results and live-dead markers
            finalReport(label_ids, cids_test, class_count, class_viability, folder_test, 'UNET')
            saveasSpectrum(label_ids, cids_test, class_count, class_viability, folder_test, 'UNET')

            print("U-NET complete")
        elif analysis_type["model_type"] == 0:
            # step 1.0 background preprocess and npv convert
            imageTonpv(folder_test, 'watershed')

            # step 2.0 load model weight and predict
            label_ids, cids_test, class_count = cnn_class(folder_test, 'watershed', 'COUNTING')
            label_ids, cids_test, class_viability = cnn_class(folder_test, 'watershed', 'VIABILITY')

            # step 3.0 save final csv results and live-dead markers
            finalReport(label_ids, cids_test, class_count, class_viability, folder_test, 'watershed')
            saveasSpectrum(label_ids, cids_test, class_count, class_viability, folder_test, 'WATERSHED')
            print("watershed complete")
        elif analysis_type["model_type"] == 2:
            # step 1.0 background preprocess and npv convert
            imageTonpv(folder_test, 'UW')

            # step 2.0 load model weight and predict
            label_ids, cids_test, class_count = cnn_class(folder_test, 'UW', 'COUNTING')
            label_ids, cids_test, class_viability = cnn_class(folder_test, 'UW', 'VIABILITY')

            # step 3.0 save final csv results and live-dead markers
            finalReport(label_ids, cids_test, class_count, class_viability, folder_test, 'UW')
            saveasSpectrum(label_ids, cids_test, class_count, class_viability, folder_test, 'UW')

            print("U-NET watershed complete")

    print("classify complete")