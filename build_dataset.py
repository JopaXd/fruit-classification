from sklearn.model_selection import train_test_split
from pyimagesearch.io import HDF5DatasetWriter
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from sklearn.preprocessing import LabelEncoder
import imutils
import cv2
import numpy as np
import os
import json

train_dataset = "/media/pavle/HDD_disk/deep-learning/fruit-recognition/fruits-360_dataset/fruits-360/Training/"
test_dataset = "/media/pavle/HDD_disk/deep-learning/fruit-recognition/fruits-360_dataset/fruits-360/Test/"
dataset_path = "/media/pavle/HDD_disk/deep-learning/fruit-recognition/fruits-360_dataset/fruits-360/"

data = []
labels = []

testX = []
testY = []

for fldr in os.listdir(train_dataset):
	for img in os.listdir(f"{train_dataset}/{fldr}"):
		data.append(f"{train_dataset}/{fldr}/{img}")
		labels.append(fldr)

for fldr in os.listdir(test_dataset):
	for img in os.listdir(f"{test_dataset}/{fldr}"):
		testX.append(f"{test_dataset}/{fldr}/{img}")
		testY.append(fldr)

(trainX, valX, trainY, valY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainWriter = HDF5DatasetWriter((len(trainX), 100, 100, 3), f"{dataset_path}/train.hdf5")
valWriter = HDF5DatasetWriter((len(valX), 100, 100, 3), f"{dataset_path}/validate.hdf5")
testWriter = HDF5DatasetWriter((len(testX), 100, 100, 3), f"{dataset_path}/test.hdf5")

le = LabelEncoder()
trainY = le.fit_transform(trainY)
valY = le.fit_transform(valY)
testY = le.fit_transform(testY)

(R, G, B) = ([], [], [])

for (path, label) in zip(trainX, trainY):
	img = cv2.imread(path)
	img = cv2.resize(img, (100,100), interpolation=cv2.INTER_AREA)
	(b, g, r) = cv2.mean(img)[:3]
	R.append(r)
	G.append(g)
	B.append(b)
	trainWriter.add([img], [label])

print("Done with the training dataset.")

for (path, label) in zip(valX, valY):
	img = cv2.imread(path)
	img = cv2.resize(img, (100,100), interpolation=cv2.INTER_AREA)
	valWriter.add([img], [label])

print("Done with the validation dataset.")

for (path, label) in zip(testX, testY):
	img = cv2.imread(path)
	img = cv2.resize(img, (100,100), interpolation=cv2.INTER_AREA)
	testWriter.add([img], [label])

print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open("means.json", "w")
f.write(json.dumps(D))
f.close()

print("Done!")