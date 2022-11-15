import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
import os
import scipy
import tensorflow as tf


## data validation test

# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

# emotion_key = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
#
# raw_data_csv_file_name = "fer2013/fer2013.csv"
# raw_data = pd.read_csv(raw_data_csv_file_name)
#
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#
#     x_pixels = np.array(raw_data["pixels"][i].split(" "), 'float32')
#     x_pixels /= 255
#     x_reshaped = x_pixels.reshape(48, 48)
#
#     plt.imshow(x_reshaped, cmap="gray", interpolation="nearest")
#     plt.xlabel(emotion_key[int(raw_data["emotion"][i])])
# plt.show()

train = "train"
val = "test"

trainingDataGenerator = ImageDataGenerator(rescale=1./255)
trainingGenerator = trainingDataGenerator.flow_from_directory(
    train,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale", # grayscale to simplify it
    class_mode="categorical" # there are 6 outputs for the 6 emotions, so categorical is best
)

validationDataGenerator = ImageDataGenerator(rescale=1/.255)
validationGenerator = validationDataGenerator.flow_from_directory(
    val,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

# define convolutional neural network model

emotRecModel = Sequential()
emotRecModel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))            # common base, 32 filters, 3x3 kernel size, > 0 outputs
emotRecModel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotRecModel.add(MaxPooling2D(pool_size=(2, 2)))                                                        # reduce spatial dimensions
emotRecModel.add(Dropout(0.25))                                                                         # prevents overfitting

emotRecModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotRecModel.add(MaxPooling2D(pool_size=(2, 2)))
emotRecModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotRecModel.add(MaxPooling2D(pool_size=(2, 2)))
emotRecModel.add(Dropout(0.25))

emotRecModel.add(Flatten())                                                                             # flattens into vector
emotRecModel.add(Dense(1024, activation='relu'))
emotRecModel.add(Dropout(0.5))

emotRecModel.add(Dense(7, activation="softmax"))

plot_model(emotRecModel, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
emotRecModel.summary()

# train model

emotRecModel.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),
    metrics=["accuracy"]
)
emotRecModelInfo = emotRecModel.fit(
    trainingGenerator,
    steps_per_epoch=28709//64,
    epochs=5,
    validation_data=validationGenerator,
    validation_steps=7178//64
)

# save model
emotRecModel.save('model.h5')





