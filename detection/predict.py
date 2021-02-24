from keras.models import Model, load_model
import numpy as np
from skimage.transform import resize
from detection.metrics.mean_iou import mean_iou
import os
from detection.encoding.rle import prob_to_rles, prob_to_rles_img
import pandas as pd
import cv2

def predict(folder_test = ''):

    # Load Preprocessing files
    X_train = np.load('detection/npv/X_train.npy')
    Y_train = np.load('detection/npv/Y_train.npy')
    X_test = np.load(folder_test + '/detection_output/NPV/X_test.npy')
    sizes_test = np.load(folder_test + '/detection_output/NPV/sizes_test.npy')

    # Predict on train, val and test
    model = load_model('detection/models/models-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
    preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
    preds_test = model.predict(X_test, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))


    TEST_PATH = folder_test + '/detection/'
    # Get train and test IDs
    test_ids = next(os.walk(TEST_PATH))[1]

    from skimage.io import imsave
    if not os.path.exists(folder_test + '/detection_output/'):
            os.mkdir(folder_test + '/detection_output/')
    if not os.path.exists(folder_test + '/detection_output/results/'):
            os.mkdir(folder_test + '/detection_output/results/')
    pred_dir = folder_test + '/detection_output/'

    # save unsampled image
    for image, imageID in zip(preds_test_upsampled, test_ids):
        image = (image * 255.).astype(np.uint8)
        # imsave(os.path.join(pred_dir, str(imageID) + '.png'), image)
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(pred_dir + 'sub-dsbowl2018-1.csv', index=False)

    # save split image
    new_test_ids_img = []
    rles_img = []
    for n, id_ in enumerate(test_ids):
        rle_img = list(prob_to_rles_img(preds_test_upsampled[n]))
        rles_img.extend(rle_img)
        new_test_ids_img.extend([id_] * len(rle_img))

    i = 0
    for image, imageID in zip(rles_img,new_test_ids_img):
        i = i + 1
        image = (image * 255.).astype(np.uint8)
        # split
        x, y, w, h = cv2.boundingRect(image)
        imsave(os.path.join(folder_test + '/detection_output/results/' + str(imageID) + '_' + str(i) + '.png'), image)
        imsave(os.path.join(folder_test + '/detection/' + (imageID) + '/masks/' + str(imageID)+ '_' +str(i) + '.png'), image)




