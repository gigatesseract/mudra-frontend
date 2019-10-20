from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
import numpy as np
import os
import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
import cv2
import matplotlib
from matplotlib import pyplot as plt


from keras import backend as K

if K.backend() == "tensorflow":
    import tensorflow
else:
    import theano


"""Ideally we should have changed image dim ordering based on Theano or Tensorflow, but for some reason I get following error when I switch it to 'tf' for Tensorflow.
	However, the outcome of the prediction doesnt seem to get affected due to this and Tensorflow gives me similar result as Theano.
	I didnt spend much time on this behavior, but if someone has answer to this then please do comment and let me know.
    ValueError: Negative dimension size caused by subtracting 3 from 1 for 'conv2d_1/convolution' (op: 'Conv2D') with input shapes: [?,1,200,200], [3,3,200,32].
"""
K.set_image_dim_ordering("th")


# Variables and declarations
img_rows, img_cols = 200, 200
img_channels = 1
batch_size = 32
nb_classes = 6
nb_epoch = 5  # 25

minValue = 70
WeightFileName = []
jsonarray = {}
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

# Path for dataset (Custom gestures go into ./gestures )
path1 = ""
framecount=0

path2 = "./dataset"


output = ["RIGHT", "OK", "NOTHING", "PEACE", "PUNCH", "STOP"]


def skinMask(frame, x0, y0, width, height, framecount, saveImg = False):
    # global guessGesture, visualize, mod, saveImg
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])

    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0, 255, 0), 1)
    # roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = frame[y0:y0+height, x0:x0+width]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)

    mask = cv2.erode(mask, skinkernel, iterations=1)
    mask = cv2.dilate(mask, skinkernel, iterations=1)

    # blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)

    # bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask=mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    if saveImg == True:
        saveROIImg(res)
        print("Gray Image dataset for gesture " +
                      gestname + " is done!! \n")
    # elif guessGesture == True and (framecount % 5) == 4:
    #     # res = cv2.UMat.get(res)
    #     t = threading.Thread(target=myNN.guessGesture, args=[mod, res])
    #     t.start()
    # elif visualize == True:
    #     layer = int(input("Enter which layer to visualize "))
    #     cv2.waitKey(0)
    #     myNN.visualizeLayers(mod, res, layer)
    #     visualize = False

    return res


def binaryMask(frame, x0, y0, width, height, framecount, saveImg = False):
    # global guessGesture, visualize, mod, saveImg
    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0, 255, 0), 1)
    roi = frame[y0:y0+height, x0:x0+width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(
        th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    if saveImg == True:
        saveROIImg(res)
    # elif guessGesture == True and (framecount % 5) == 4:
    #     # ores = cv2.UMat.get(res)
    #     t = threading.Thread(target=myNN.guessGesture, args=[mod, res])
    #     t.start()
    # elif visualize == True:
    #     layer = int(input("Enter which layer to visualize "))
    #     cv2.waitKey(1)
    #     myNN.visualizeLayers(mod, res, layer)
    #     visualize = False

    return res

def bkgrndSubMask(frame, x0, y0, width, height, framecount, saveImg = False):
    # global guessGesture, takebkgrndSubMask, visualize, mod, bkgrnd, saveImg

    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0, 255, 0), 1)
    roi = frame[y0:y0+height, x0:x0+width]
    # roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Take background image
    # if takebkgrndSubMask == True:
    bkgrnd = roi
        # takebkgrndSubMask = False
        # print("Refreshing background image for mask...")

    # Take a diff between roi & bkgrnd image contents
    diff = cv2.absdiff(roi, bkgrnd)

    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    mask = cv2.GaussianBlur(diff, (3, 3), 5)
    mask = cv2.erode(diff, skinkernel, iterations=1)
    mask = cv2.dilate(diff, skinkernel, iterations=1)
    res = cv2.bitwise_and(roi, roi, mask=mask)

    if saveImg == True:
        saveROIImg(res)
    # elif guessGesture == True and (framecount % 5) == 4:
    #     t = threading.Thread(target=myNN.guessGesture, args=[mod, res])
    #     t.start()
    #     # t.join()
    #     # myNN.update(plot)

    # elif visualize == True:
    #     layer = int(input("Enter which layer to visualize "))
    #     cv2.waitKey(0)
    #     myNN.visualizeLayers(mod, res, layer)
    #     visualize = False

    return res

def update(plot):
    global jsonarray
    h = 450
    y = 30
    w = 45
    font = cv2.FONT_HERSHEY_SIMPLEX

    for items in jsonarray:
        mul = (jsonarray[items]) / 100
        cv2.line(plot, (0, y), (int(h * mul), y), (255, 0, 0), w)
        cv2.putText(plot, items, (0, y + 5), font, 0.5, (0, 255, 0), 1, 1)
        y = y + w + 30

    return plot

def debugme():
    import pdb

    pdb.set_trace()


def convertToGrayImg(path1, path2):
    listing = os.listdir(path1)
    for file in listing:
        if file.startswith("."):
            continue
        img = Image.open(path1 + "/" + file)
        # img = img.resize((img_rows,img_cols))
        grayimg = img.convert("L")
        grayimg.save("./gestureRecog/dataset/" +  file, "PNG")



def preprocess(frame, bgrnd=False, binary_mask = False):
    global framecount
    x0 = 400
    y0 = 200
    height = 200
    width = 200
    frame = cv2.flip(frame, 3)
    frame = cv2.resize(frame, (640, 480))
    if bgrnd == True:
        roi = bkgrndSubMask(frame, x0, y0, width,
                                    height, framecount)
    elif binary_mask == True:
        roi = binaryMask(frame, x0, y0, width,
                            height, framecount)
    else:
        roi = skinMask(frame, x0, y0, width, height, framecount)
    framecount+=1
    cv2.imshow('ROI', roi)
    cv2.imshow('Original', frame)
    return roi
# %%
def modlistdir(path, pattern=None):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        # This check is to ignore any hidden files/folders
        if pattern == None:
            if name.startswith("."):
                continue
            else:
                retlist.append(name)
        elif name.endswith(pattern):
            retlist.append(name)

    return retlist


# Load CNN model
def loadCNN(with_weights=True):
    global get_output
    model = Sequential()

    model.add(
        Conv2D(
            nb_filters,
            (nb_conv, nb_conv),
            padding="valid",
            input_shape=(img_channels, img_rows, img_cols),
        )
    )
    convout1 = Activation("relu")
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation("relu")
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
    )

    # Model summary
    # model.summary()
    # Model conig details
    # model.get_config()
    if with_weights:
        model.load_weights("./gestureRecog/pullingo.hdf5", by_name=True)

    layer = model.layers[11]
    get_output = K.function(
        [model.layers[0].input, K.learning_phase()], [layer.output])

    return model


# This function does the guessing work based on input images
def guessGesture(model, img):
    global output, get_output, jsonarray
    # Load image and flatten it
    image = np.array(img).flatten()

    # reshape it
    image = image.reshape(img_channels, img_rows, img_cols)

    # float32
    image = image.astype("float32")

    # normalize it
    image = image / 255

    # reshape for NN
    rimage = image.reshape(1, img_channels, img_rows, img_cols)

    # Now feed it to the NN, to fetch the predictions
    # index = model.predict_classes(rimage)
    # prob_array = model.predict_proba(rimage)

    prob_array = get_output([rimage, 0])[0]

    # print prob_array

    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1

    # Get the output with maximum probability
    import operator

    guess = max(d.items(), key=operator.itemgetter(1))[0]
    prob = d[guess]

    if prob > 60.0:
        # print(guess + "  Probability: ", prob)

        # Enable this to save the predictions in a json file,
        # Which can be read by plotter app to plot bar graph
        # dump to the JSON contents to the file

        # with open('gesturejson.txt', 'w') as outfile:
        #    json.dump(d, outfile)
        jsonarray = d

        return output.index(guess)

    else:
        # Lets return index 1 for 'Nothing'
        return 5


def predict(frame):
    model = loadCNN()
    return guessGesture(model, frame)
# %%
def initializers():
    imlist = modlistdir(path2)

    # open one image to get size
    image1 = np.array(Image.open(path2 + "/" + imlist[0]))
    # plt.imshow(im1)

    m, n = image1.shape[0:2]  # get the size of the images
    total_images = len(imlist)  # get the 'total' number of images

    # create matrix to store all flattened images
    immatrix = np.array(
        [
            np.array(Image.open(path2 + "/" + images).convert("L")).flatten()
            for images in sorted(imlist)
        ],
        dtype="f",
    )

    print(immatrix.shape)

    input("Press any key")

    #########################################################
    # Label the set of images per respective gesture type.
    ##
    label = np.ones((total_images,), dtype=int)

    samples_per_class = int(total_images / nb_classes)
    print("samples_per_class - ", samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class

    """
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    """

    data, Label = shuffle(immatrix, label, random_state=2)
    train_data = [data, Label]

    (X, y) = (train_data[0], train_data[1])

    # Split X and y into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4
    )

    X_train = X_train.reshape(
        X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # normalize
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test


def trainModel(model):

    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers()

    # Now start the training of the loaded model
    hist = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_split=0.2,
    )

    visualizeHis(hist)

    ans = input("Do you want to save the trained weights - y/n ?")
    if ans == "y":
        filename = input("Enter file name - ")
        fname = path + str(filename) + ".hdf5"
        model.save_weights(fname, overwrite=True)
    else:
        model.save_weights("newWeight.hdf5", overwrite=True)

    # Save model as well
    # model.save("newModel.hdf5")


# %%


def visualizeHis(hist):
    # visualizing losses and accuracy

    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    train_acc = hist.history["acc"]
    val_acc = hist.history["val_acc"]
    xc = range(nb_epoch)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel("num of Epochs")
    plt.ylabel("loss")
    plt.title("train_loss vs val_loss")
    plt.grid(True)
    plt.legend(["train", "val"])
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    # plt.style.use(['classic'])

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel("num of Epochs")
    plt.ylabel("accuracy")
    plt.title("train_acc vs val_acc")
    plt.grid(True)
    plt.legend(["train", "val"], loc=4)

    plt.show()


# %%
def visualizeLayers(model):
    imlist = modlistdir("./imgs")
    if len(imlist) == 0:
        print("Error: No sample image file found under './imgs' folder.")
        return
    else:
        print("Found these sample image files - {}".format(imlist))

    img = int(
        input(
            "Which sample image file to load (enter the INDEX of it, which starts from 0): "
        )
    )
    layerIndex = int(
        input(
            "Enter which layer to visualize. Enter -1 to visualize all layers possible: "
        )
    )

    if img <= len(imlist):

        image = np.array(Image.open(
            "./imgs/" + imlist[img]).convert("L")).flatten()

        # Predict
        print("Guessed Gesture is {}".format(
            output[guessGesture(model, image)]))

        # reshape it
        image = image.reshape(img_channels, img_rows, img_cols)

        # float32
        image = image.astype("float32")

        # normalize it
        image = image / 255

        # reshape for NN
        input_image = image.reshape(1, img_channels, img_rows, img_cols)
    else:
        print("Wrong file index entered !!")
        return

    # visualizing intermediate layers
    # output_layer = model.layers[layerIndex].output
    # output_fn = theano.function([model.layers[0].input], output_layer)
    # output_image = output_fn(input_image)

    if layerIndex >= 1:
        visualizeLayer(model, img, input_image, layerIndex)
    else:
        tlayers = len(model.layers[:])
        print("Total layers - {}".format(tlayers))
        for i in range(1, tlayers):
            visualizeLayer(model, img, input_image, i)


# %%
def visualizeLayer(model, img, input_image, layerIndex):

    layer = model.layers[layerIndex]

    get_activations = K.function(
        [model.layers[0].input, K.learning_phase()], [layer.output]
    )
    activations = get_activations([input_image, 0])[0]
    output_image = activations

    # If 4 dimensional then take the last dimension value as it would be no of filters
    if output_image.ndim == 4:
        # Rearrange dimension so we can plot the result
        # o1 = np.rollaxis(output_image, 3, 1)
        # output_image = np.rollaxis(o1, 3, 1)
        output_image = np.moveaxis(output_image, 1, 3)

        print(
            "Dumping filter data of layer{} - {}".format(
                layerIndex, layer.__class__.__name__
            )
        )
        filters = len(output_image[0, 0, 0, :])

        fig = plt.figure(figsize=(8, 8))
        # This loop will plot the 32 filter data for the input image
        for i in range(filters):
            ax = fig.add_subplot(6, 6, i + 1)
            # ax.imshow(output_image[img,:,:,i],interpolation='none' ) #to see the first filter
            ax.imshow(output_image[0, :, :, i], "gray")
            # ax.set_title("Feature map of layer#{} \ncalled '{}' \nof type {} ".format(layerIndex,
            #                layer.name,layer.__class__.__name__))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        plt.tight_layout()
        # plt.show()
        savedfilename = (
            "img_"
            + str(img)
            + "_layer"
            + str(layerIndex)
            + "_"
            + layer.__class__.__name__
            + ".png"
        )
        fig.savefig(savedfilename)
        print("Create file - {}".format(savedfilename))
        # plt.close(fig)
    else:
        print(
            "Can't dump data of this layer{}- {}".format(
                layerIndex, layer.__class__.__name__
            )
        )
