# USAGE
# python train_model.py --dataset dataset

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import os

# construct the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to the output plot")
ap.add_argument("-o", "--output", type=str,
	default="facemask_model.model",
	help="path to the output model")
args = vars(ap.parse_args())

#learning rate
L_RATE = 1e-4 
#epochs (the number times that the learning algorithm will work through the entire training dataset)
EPOCHS = 20
#batch size (the number of training examples utilized in one iteration) (hardware dependent)
BS = 32

#initialize the list of paths of the images in the dataset 
imagePaths = list(paths.list_images(args["dataset"]))

data = [] 
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the lable from the path name (with_mask/without_mask)
	label = imagePath.split(os.path.sep)[-2]

	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# add the data and respective labels to separate lists
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#load the pretrained MobileNetV2 model weights
mobilenet = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the classifier that will be placed on top of the mobilenet model
classifier = mobilenet.output
classifier = AveragePooling2D(pool_size=(7, 7))(classifier)
classifier = Flatten(name="flatten")(classifier)
classifier = Dense(128, activation="relu")(classifier)
classifier = Dropout(0.5)(classifier)
classifier = Dense(2, activation="softmax")(classifier)

# the finite model will consist of the combination between the mobilenetmodel
# and the trainable classifier
model = Model(inputs=mobilenet.input, outputs=classifier)

# freeze the mobile net weights so they are not updated during the training 
# of the classifier (those weights are already trained)
for layer in mobilenet.layers:
	layer.trainable = False

#use the Adam optimiser 
opt = Adam(lr=L_RATE, decay=L_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# training the classifier
Mtrain = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)
	

# saving the model 
model.save(args["output"], save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), Mtrain.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), Mtrain.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), Mtrain.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), Mtrain.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
