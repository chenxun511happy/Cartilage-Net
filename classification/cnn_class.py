import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import interp
import csv
from keras.models import Model, load_model
import os

class CartilageNet:
    def __init__(self):
        self.X = None
        self.test_X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_data_size = None
        self.weights = None
        self.model = None
        self.nb_classes = None
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.n_gpus = 8
        

    def split_data(self, X, test_data_size, category_quant =[]):
        """
        Split data into test and training data sets.

        INPUT
            y_file_path: path to CSV containing labels
            X: NumPy array of arrays
            test_data_size: size of test/train split. Value from 0 to 1

        OUTPUT
            Four arrays: X_train, X_test, y_train, and y_test
        """
        # # labels = pd.read_csv(y_file_path, nrows=60)
        # labels = pd.read_csv(y_file_path)
        self.X = np.load(X)
        # self.y = np.array(labels['level'])
        self.y = np.array([0 for i in range(category_quant[0])])
        self.y = np.append(self.y, np.array([1 for i in range(category_quant[1])]), axis=0)
        self.y = np.append(self.y, np.array([2 for i in range(category_quant[2])]), axis=0)
        self.y = np.append(self.y, np.array([3 for i in range(category_quant[3])]), axis=0)
        self.y = np.append(self.y, np.array([4 for i in range(category_quant[4])]), axis=0)

        if (self.X.shape[0] != self.y.shape[0]):
            raise('array no equal')

        self.weights = class_weight.compute_class_weight('balanced', np.unique(self.y), self.y)
        self.test_data_size = test_data_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_data_size,
                                                                                random_state=42)
    def split_data_test(self, y_file_path, X, test_data_size, testlabel):
        """
        Split data into test and training data sets.

        INPUT
            y_file_path: path to CSV containing labels
            X: NumPy array of arrays
            test_data_size: size of test/train split. Value from 0 to 1

        OUTPUT
            Four arrays: X_train, X_test, y_train, and y_test
        """
        # labels = pd.read_csv(y_file_path, nrows=60)
        labels = pd.read_csv(y_file_path, nrows =4, skiprows = range(1, (testlabel)*4))
        test_X = np.load(X)
        self.X = test_X[testlabel*4:testlabel*4+4]
        self.y = np.array(labels['level'])
        self.weights = class_weight.compute_class_weight('balanced', np.unique(self.y), self.y)
        self.test_data_size = test_data_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_data_size,
                                                                                random_state=42,shuffle =False )        

    def reshape_data(self, img_rows, img_cols, channels, nb_classes):
        """
        Reshapes arrays into format for MXNet

        INPUT
            img_rows: Array (image) height
            img_cols: Array (image) width
            channels: Specify if image is grayscale(1) or RGB (3)
            nb_classes: number of image classes/ categories

        OUTPUT
            Reshaped array of NumPy arrays
        """
        self.nb_classes = nb_classes
        self.X_train = self.X_train.reshape(self.X_train.shape[0], img_rows, img_cols, channels)
        self.X_train = self.X_train.astype("float32")
        self.X_train /= 255

        self.y_train = np_utils.to_categorical(self.y_train, self.nb_classes)

        self.X_test = self.X_test.reshape(self.X_test.shape[0], img_rows, img_cols, channels)
        self.X_test = self.X_test.astype("float32")
        self.X_test /= 255

        self.y_test = np_utils.to_categorical(self.y_test, self.nb_classes)

        print("X_train Shape: ", self.X_train.shape)
        print("X_test Shape: ", self.X_test.shape)
        print("y_train Shape: ", self.y_train.shape)
        print("y_test Shape: ", self.y_test.shape)

    def cnn_model_five(self, nb_filters, kernel_size, batch_size, nb_epoch):
        """
        Define and run the convolutional neural network


        """

        self.model = Sequential()
        self.model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                              padding="valid",
                              strides=1,
                              input_shape=(self.img_rows, self.img_cols, self.channels), activation="relu"))

        self.model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

        self.model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

        self.model.add(MaxPooling2D(pool_size=(8, 8)))

        self.model.add(Flatten())
        print("Model flattened out to: ", self.model.output_shape)

        self.model.add(Dense(2048, activation="relu"))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(2048, activation="relu"))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(self.nb_classes, activation="softmax"))

#        self.model = multi_gpu_model(self.model, gpus=self.n_gpus)

        self.model.compile(loss="categorical_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

        stop = EarlyStopping(monitor="acc", min_delta=0,
                             patience=50,
                             mode="auto")

        self.model.fit(self.X_train, self.y_train, batch_size=batch_size,
                       epochs=nb_epoch,
                       verbose=1,
                       validation_split=0.2,
                       class_weight=self.weights,
                       callbacks=[stop])

        return self.model

    
    def predict(self):
        """
        Predicts the model output, and computes precision, recall, and F1 score.

        INPUT
            model: Model trained in Keras

        OUTPUT
            Precision, Recall, and F1 score
        """
        predictions = self.model.predict(self.X_test)
        print("Pre-y: ", predictions)
        print("Test-y: ", self.y_test)
        
        
        
        n_classes = 5
        fpr = [0 for col in range(n_classes)]
        tpr = [0 for col in range(n_classes)]
        thresholds = [0 for col in range(n_classes)]
        aucmean = [0 for col in range(n_classes)]
        
        
        for i in range(n_classes):
          fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(self.y_test[:, i], predictions[:, i])
          aucmean[i] = metrics.auc(fpr[i], tpr[i])
        
        aucmean_total = np.average(aucmean)

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes
        
        
        plt.figure()
        lw = 2
        plt.plot(all_fpr, mean_tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.savefig('data/figROC.png')
        plt.show()
        
        
        
        predictions = np.argmax(predictions, axis=1)

        # predictions[predictions >=1] = 1 # Remove when non binary classifier

        self.y_test = np.argmax(self.y_test, axis=1)
        
        print ('auc mean', aucmean_total)
        confusion_mat = confusion_matrix(self.y_test, predictions)
        
        FP = confusion_mat.sum(axis=0) - np.diag(confusion_mat)  
        FN = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
        TP = np.diag(confusion_mat)
        TN = confusion_mat[:].sum() - (FP + FN + TP)
        
        
        
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        print ('Sensitivity', np.average(TPR))
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        print ('Specificity', np.average(TNR))
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        print ('Positive predictive value', np.average(PPV) )
        # Negative predictive value
        NPV = TN/(TN+FN)
        print ('Negative predictive value', np.average(NPV) )

        precision = precision_score(self.y_test, predictions, average="micro")
        recall = recall_score(self.y_test, predictions, average="micro")
        f1 = f1_score(self.y_test, predictions, average="micro")
        cohen_kappa = cohen_kappa_score(self.y_test, predictions)
        quad_kappa = 0
        return precision, recall, f1, cohen_kappa, quad_kappa

    def save_model(self, score, model_name):
        """
        Saves the model, based on scoring criteria input.

        INPUT
            score: Scoring metric used to save model or not.
            model_name: name for the model to be saved.

        OUTPUT
            Saved model, based on scoring criteria input.
        """
        if score >= 0.7:
            print("Saving Model")
            self.model.save("models/UW4.h5")
        else:
            print("Model Not Saved. Score: ", score)


def cnn_class(path = '', MODEL_TYPE = 'UNET', CLASS_TYPE = 'COUNTING'):
    cnn = CartilageNet()
    category_quant = []
    if MODEL_TYPE == 'UNET' and CLASS_TYPE == 'COUNTING':
       npv_path = 'classification/npv/COUNTING/UNET/X_train_Unet_counting.npy'
       category_quant = [600, 1656, 1576, 424, 512]
       imgs_test = np.load('classification/npv/COUNTING/UNET/X_test_Unet_counting.npy')
       h5_path = 'classification/models/COUNTING/UNET/Unet4.h5'
       ids_test = np.load('classification/npv/COUNTING/UNET/X_test_id.npy')
       cids_test = np.load('classification/npv/COUNTING/UNET/X_test_cid.npy')

    elif MODEL_TYPE == 'UW' and CLASS_TYPE == 'COUNTING':
       npv_path = 'classification/npv/COUNTING/UW/X_train_UW_counting.npy'
       category_quant = [824, 3336, 1528, 384, 184]
       imgs_test = np.load('classification/npv/COUNTING/UW/X_test_UW_counting.npy')
       h5_path = 'classification/models/COUNTING/UW/UW4.h5'
       ids_test = np.load('classification/npv/COUNTING/UW/X_test_id.npy')
       cids_test = np.load('classification/npv/COUNTING/UW/X_test_cid.npy')

    elif MODEL_TYPE == 'UNET' and CLASS_TYPE == 'VIABILITY':
        npv_path = 'classification/npv/VIABILITY/UNET/X_train_UnetViability.npy'
        category_quant = [1696, 1264, 1056, 416, 336]
        imgs_test = np.load('classification/npv/VIABILITY/UNET/X_test_UnetViability.npy')
        h5_path = 'classification/models/VIABILITY/UNET/Unet2.h5'
        ids_test = np.load('classification/npv/VIABILITY/UNET/X_test_id.npy')
        cids_test = np.load('classification/npv/VIABILITY/UNET/X_test_cid.npy')

    elif MODEL_TYPE == 'UW' and CLASS_TYPE == 'VIABILITY':
        npv_path = 'classification/npv/VIABILITY/UW/X_train_UWViability.npy'
        category_quant = [2480, 2192, 1152, 320, 112]
        imgs_test = np.load('classification/npv/VIABILITY/UW/X_test_UWViability.npy')
        h5_path = 'classification/models/VIABILITY/UW/UW1.h5'
        ids_test = np.load('classification/npv/VIABILITY/UW/X_test_id.npy')
        cids_test = np.load('classification/npv/VIABILITY/UW/X_test_cid.npy')

    elif MODEL_TYPE == 'watershed' and CLASS_TYPE == 'COUNTING':
        npv_path = 'classification/npv/COUNTING/WATERSHED/X_train_WATER_counting.npy'
        category_quant = [2480, 2192, 1152, 320, 112]
        imgs_test = np.load('classification/npv/COUNTING/WATERSHED/X_test_WATER_counting.npy')
        h5_path = 'classification/models/COUNTING/WATERSHED/DR_Class_Countwatershed_recall_0.8138.h5'
        ids_test = np.load('classification/npv/COUNTING/WATERSHED/X_test_id.npy')
        cids_test = np.load('classification/npv/COUNTING/WATERSHED/X_test_cid.npy')

    elif MODEL_TYPE == 'watershed' and CLASS_TYPE == 'VIABILITY':
        npv_path = 'classification/npv/VIABILITY/WATERSHED/X_train_WATERViability.npy'
        category_quant = [2480, 2192, 1152, 320, 112]
        imgs_test = np.load('classification/npv/VIABILITY/WATERSHED/X_test_WATERViability.npy')
        h5_path = 'classification/models/VIABILITY/WATERSHED/DR_Class_Viabilitywatershed_recall_0.8336.h5'
        ids_test = np.load('classification/npv/VIABILITY/WATERSHED/X_test_id.npy')
        cids_test = np.load('classification/npv/VIABILITY/WATERSHED/X_test_cid.npy')
    else:
        raise ValueError("Something Wrong")

    # cnn.split_data(X=npv_path, test_data_size=0.2, category_quant= category_quant)
    # cnn.reshape_data(img_rows=256, img_cols=256, channels=3, nb_classes=5)
    # #apply model

    # model = load_weights('models/UW4.h5')
    model = load_model(h5_path)
    print("--------Load weight complete------")

    imgs_test = imgs_test.reshape(imgs_test.shape[0], 256, 256, 3)
    imgs_test= imgs_test.astype("float32")
    imgs_test /= 255

    class_test = model.predict(imgs_test)
    print("Class_test: ", class_test)
    class_test_count = np.argmax(class_test, axis=1)
    print("total cell numbers", class_test_count)

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(path + '/classification/'):
        os.mkdir(path + '/classification/')

    label_ids = []
    for index, ids in enumerate(ids_test):
        label_ids.append(str(format(ids, '05d')) + '_' + str(cids_test[index]))
    out = open(path + '/classification/' + str(timestr) + MODEL_TYPE + '_' + CLASS_TYPE + 'CNN_csv.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(label_ids)
    csv_write.writerow(class_test_count)

    return ids_test, cids_test, class_test_count