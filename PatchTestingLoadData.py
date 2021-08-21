img_height = 128 #128 is being upscaled cuz resnet50 needs minimum 197
img_width = img_height
batch_size = 45
# nb_epochs = 300
# patience_val = 10
# model_name = "Kather_ResNet50_with_SGD_Augment_lr0_01"
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
from imutils import paths
# A function to load data from a given directory
def load_data(data_dir):
    data = []
    labels = []
    class_dirs = os.listdir(data_dir)

    for direc in class_dirs:
        class_dir = os.path.join(data_dir, direc)
        for imagepath in tqdm(list(paths.list_images(class_dir))):
            image = cv2.imread(imagepath)
            image = cv2.resize(image, (img_width, img_height))  # incase images not of same size
            data.append(image)
            labels.append(direc)
    # normalizing and converting to numpy array format
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)
    return data, labels

import pandas as pd
import numpy as np

test_dir = "E:/MyPipeline/training_data/valid/"
# test_dir = "D:/Downloads/NCT-CRC-HE-100K/Testing/"

save_path = 'E:/MyPipeline/training_data/Results/'
# model.load_weights(save_path+"_"+model_name+"_transfer_trained_wts.h5")
from tensorflow.python import keras
model = keras.models.load_model(save_path+"DatasetSKMC_InceptionResNetV2_with_SGDM_lr0_0003_best_model_val_acc_max.h5")
print('loading test images')
X_test, y_test = load_data(test_dir)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_test)

predictions = model.predict(X_test, verbose=1)

print("predictions: ", predictions)


pd.DataFrame(predictions).to_csv("E:/MyPipeline/training_data/Results/Apni_predictions.csv")

y_pred = np.argmax(predictions, axis=1)
mask = y_pred==y_test
correct = np.count_nonzero(mask)
print ("Correct percentage", correct*100.0/y_pred.size)

print("True: ", y_test)
print("Predicted: ", y_pred)
print("Mask: ", mask)
print("Correct: ", correct)
print("total size: ", y_pred.size)

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

classes = ["0_SMFR", "1_Non_ROI", "2_Tumour", "3_Lymphocytes", "4_Nerve"]
# classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
print(classification_report(y_test, y_pred, target_names=classes))

conf = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf,
                              display_labels=classes)
disp.plot()
plt.savefig(save_path + 'DatasetSKMC_VGG19_ConfusionMatrixImgLoad.png')
plt.show()
