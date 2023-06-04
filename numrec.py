from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

mnist = fetch_openml("mnist_784")
# The data needs to be reshaped, now it is 70,000 x 28 x 28 x 1.
# the 28 exists since that is the height and width of each image.
# If the data were color, rather than black and white, it would be: 70,000 x 28 x 28 x 3
mnist.data = np.array(mnist.data).reshape(-1, 28, 28, 1) / 255
mnist.target = mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=0.2, random_state=2
)
from google.colab import drive

drive.mount("/content/drive")
import numpy as np
from PIL import Image
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert("L")
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new("L", (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(
            round((20.0 / width * height), 0)
        )  # resize height according to ratio width
        if nheight == 0:  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(
            round((20.0 / height * width), 0)
        )  # resize width according to ratio height
        if nwidth == 0:  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva


import matplotlib.pyplot as plt

inp = imageprepare("/content/drive/MyDrive/intro to ML/smallthe.png")

print(inp)


# Solution for a simple
from keras import layers
from keras import models
from keras import optimizers
from sklearn.metrics import accuracy_score


class MNISTClassifier:

    """
    This is the constructor that we call every time we make a new predictor.
    The inputs are the training set input and output features.
    """

    def __init__(self, X_train, y_train):
        self.cnn = models.Sequential()

        # network structure
        self.cnn.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
        self.cnn.add(layers.MaxPool2D(pool_size=(2, 2)))
        self.cnn.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
        self.cnn.add(layers.Flatten())
        self.cnn.add(layers.Dense(10, activation="softmax"))

        # gradient descent
        self.cnn.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizers.Adam(learning_rate=1e-4),
            metrics=["acc"],
        )
        self.cnn.fit(X_train, y_train, batch_size=32, epochs=2)

    """
  Predicts all the rows of X_test.
  The returned array should have length equal to the rows of X_test
  """

    def predict(self, X_test):
        return self.cnn.predict(X_test)

    """
  Evaluates the recommender on the test set and reports the MSE
  """

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test).argmax(axis=1)
        return accuracy_score(y_test, preds), preds


classifier = MNISTClassifier(X_train, y_train)
accuracy, predictions = classifier.evaluate(X_test, y_test)
print("Accuracy:", accuracy)


# Convert to numpy array
w_train = np.array(inp)
w_train.shape
w_train = np.reshape(w_train, (1, 28, 28, 1))


print(classifier.predict(w_train))

predictions = classifier.predict(w_train)[0]
plt.bar(range(10), predictions)
