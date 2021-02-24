import cv2
from os import listdir
from os.path import isfile, join
import os
from PIL import Image
mergeSize = 1024


def PositionMap(i = 0):
    x, y, w, h = 0, 0, 256, 256
    if i == 0 or i == 1 or i == 2 or i == 3:
        x, y, w, h = 0, 256*i, 256, 256
    elif i == 4 or i == 5 or i == 6 or i == 7:
        x, y, w, h = 256, 256*(i-4), 256, 256
    elif i == 8 or i == 9 or i == 10 or i == 11:
        x, y, w, h = 512, 256 * (i - 8), 256, 256
    elif i == 12 or i == 13 or i == 14 or i == 15:
        x, y, w, h = 768, 256 * (i - 12), 256, 256
    return (x, y)



if __name__ == '__main__':
    in_path = "E:/chen_AI/cartilage_3/test1024/precrop/classification"
    save_path = "E:/chen_AI/cartilage_3/test1024/postmerge/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    onlyfiles = [f for f in listdir(in_path) if isfile(join(in_path, f))]

    raw = []
    fileName = []
    for folder in onlyfiles:
      if not 'csv' in folder:
          raw.append(Image.open(in_path + '/' + folder))
          fileName.append(folder)

    mergeNum = int(mergeSize/raw[0].size[0])
    fileNum = int(len(raw)/(mergeNum*mergeNum))
    for n in range(fileNum):
        new_im = Image.new("RGB", (mergeSize, mergeSize))
        for i in range(mergeNum*mergeNum):
            if mergeSize == 1024:
                pasteBox = PositionMap(i)
            else:
                pasteBox = (0, 0)
            new_im.paste(raw[i + n * mergeNum * mergeNum], pasteBox)

        new_im.save(save_path + fileName[n * mergeNum * mergeNum].split('.')[0] + '.png')

