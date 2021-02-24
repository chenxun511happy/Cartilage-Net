# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

import numpy as np
from skimage.morphology import label

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def replacet(arr, result, T):
    shape = arr.shape
    result = np.copy(result)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            if arr[x, y] == T:
                result[x, y] = 1
            else:
                result[x, y] = 0
    return result

def prob_to_rles_img(x, cutoff=0.5):
   lab_img = label(x > cutoff)
   img = []
   for i in range(1, lab_img.max() + 1):
       img.append(replacet(lab_img, x, i))

   return img
