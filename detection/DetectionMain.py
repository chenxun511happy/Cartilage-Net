from detection.dataimport import dataImport
from detection.data import load_test_data
from detection.predict import predict
from detection.watershed import waterSHED
from detection.mask_merge import maskMerge
from detection.postprocessingUnet import RedContoursUnet
from detection.postprocessingWatershed import RedContoursWatershed
from detection.getTestSetsClassification import splitintoClusters

def detectionMain(folder_test = '', folder_train = '', analysis_type = dict()):
    if not analysis_type["predict_type"] == 0:
        if analysis_type["model_type"] == 1:
            # step 1.0 background preprocess
            dataImport(folder_test)

            # step 2.0 convert image to npv
            load_test_data(folder_test, analysis_type)

            # step 3.0 model load and predict
            predict(folder_test)
            # step 4.0 post processing save with RedContours
            maskMerge(folder_test)
            RedContoursUnet(folder_test)

            # step 5.0 split into clusters and rotation only for train sets
            splitintoClusters(folder_test, 'UNET')

        elif analysis_type["model_type"] == 0:
            waterSHED(folder_test)

            splitintoClusters(folder_test, 'watershed')
            print("watershed complete")
        elif analysis_type["model_type"] == 2:
            # step 1.0 background preprocess
            dataImport(folder_test)

            # step 2.0 convert image to npv
            load_test_data(folder_test)

            # step 3.0 model load and predict
            predict(folder_test)
            # step 4.0 post processing save with RedContours
            maskMerge(folder_test)
            RedContoursWatershed(folder_test)

            # step 5.0 split into clusters and rotation only for train sets
            splitintoClusters(folder_test, 'UW')

            print("U-NET watershed complete")
    print("detection complete")