import numpy as np
from PIL import Image
import os
import time
from os.path import isfile, join
from os import listdir

def change_image_name(df, column):
    """
    Appends the suffix '.jpeg' for all image names in the DataFrame

    INPUT
        df: Pandas DataFrame, including columns to be altered.
        column: The column that will be changed. Takes a string input.

    OUTPUT
        Pandas DataFrame, with a single column changed to include the
        aforementioned suffix.
    """
    return [i + '.jpeg' for i in df[column]]


def convert_images_to_arrays_train(file_path):
    """
    Converts each image to an array, and appends each array to a new NumPy
    array, based on the image column equaling the image file name.

    INPUT
        file_path: Specified file path for resized test and train images.
        df: Pandas DataFrame being used to assist file imports.

    OUTPUT
        NumPy array of image arrays.
    """
    img_array = np.array([np.array(Image.open(file_path + folder)) for folder in os.listdir(file_path)])

    return img_array


def save_to_array(arr_name, arr_object):
    """
    Saves data object as a NumPy file. Used for saving train and test arrays.

    INPUT
        arr_name: The name of the file you want to save.
            This input takes a directory string.
        arr_object: NumPy array of arrays. This object is saved as a NumPy file.

    OUTPUT
        NumPy array of image arrays
    """
    return np.save(arr_name, arr_object)


def imageTonpv(path = '', MODEL_TYPE = 'UNET', CLASS_TYPE = 'COUNTING'):

    start_time = time.time()
    print("Writing Train Array")
    if MODEL_TYPE == 'UNET':
        X_test = convert_images_to_arrays_train(path + '/detection_output/resized_images/')
        print(X_test.shape)
        print("Saving Train Array")
        save_to_array('classification/npv/COUNTING/UNET/X_test_Unet_counting.npy', X_test)
        save_to_array('classification/npv/VIABILITY/UNET/X_test_UnetViability.npy', X_test)
        mypath = path + '/detection_output/resized_images/'
        images = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        total = len(images)
        imgs_id = np.ndarray((total,), dtype=np.int32)
        clusters_id = np.ndarray((total,), dtype=np.int32)
        i = 0
        for image_name in images:
            img_id = int(image_name.split('_')[0])
            cluster_id = int(image_name.split('_')[1].split('.')[0])
            imgs_id[i] = img_id
            clusters_id[i] = cluster_id
            i += 1
        np.save('classification/npv/VIABILITY/UNET/X_test_id.npy', imgs_id)
        np.save('classification/npv/COUNTING/UNET/X_test_id.npy', imgs_id)
        np.save('classification/npv/VIABILITY/UNET/X_test_cid.npy', clusters_id)
        np.save('classification/npv/COUNTING/UNET/X_test_cid.npy', clusters_id)

    elif MODEL_TYPE == 'UW':
        X_test = convert_images_to_arrays_train(path + '/detection_output/UW/resized_images/')
        print(X_test.shape)
        print("Saving Train Array")
        save_to_array('classification/npv/COUNTING/UW/X_test_UW_counting.npy', X_test)
        save_to_array('classification/npv/VIABILITY/UW/X_test_UWViability.npy', X_test)

        mypath = path + '/detection_output/UW/resized_images/'
        images = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        total = len(images)
        imgs_id = np.ndarray((total,), dtype=np.int32)
        clusters_id = np.ndarray((total,), dtype=np.int32)
        i = 0
        for image_name in images:
            img_id = int(image_name.split('_')[0])
            cluster_id = int(image_name.split('_')[1].split('.')[0])
            imgs_id[i] = img_id
            clusters_id[i] = cluster_id
            i += 1
        np.save('classification/npv/VIABILITY/UW/X_test_id.npy', imgs_id)
        np.save('classification/npv/COUNTING/UW/X_test_id.npy', imgs_id)
        np.save('classification/npv/VIABILITY/UW/X_test_cid.npy', clusters_id)
        np.save('classification/npv/COUNTING/UW/X_test_cid.npy', clusters_id)
    else:
        X_test = convert_images_to_arrays_train(path + '/detection_output/watershed/resized_images/')
        print(X_test.shape)
        print("Saving Train Array")
        save_to_array('classification/npv/COUNTING/WATERSHED/X_test_WATER_counting.npy', X_test)
        save_to_array('classification/npv/VIABILITY/WATERSHED/X_test_WATERViability.npy', X_test)
        print("Saving Train Array")

        mypath = path + '/detection_output/watershed/resized_images/'
        images = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        total = len(images)
        imgs_id = np.ndarray((total,), dtype=np.int32)
        clusters_id = np.ndarray((total,), dtype=np.int32)
        i = 0
        for image_name in images:
            img_id = int(image_name.split('_')[0])
            cluster_id = int(image_name.split('_')[1].split('.')[0])
            imgs_id[i] = img_id
            clusters_id[i] = cluster_id
            i += 1
        np.save('classification/npv/VIABILITY/WATERSHED/X_test_id.npy', imgs_id)
        np.save('classification/npv/COUNTING/WATERSHED/X_test_id.npy', imgs_id)
        np.save('classification/npv/VIABILITY/WATERSHED/X_test_cid.npy', clusters_id)
        np.save('classification/npv/COUNTING/WATERSHED/X_test_cid.npy', clusters_id)

    print("--- %s seconds ---" % (time.time() - start_time))