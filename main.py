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

print(tf.__version__)


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
    color_mode="grayscale",
    class_mode="categorical"
)

validationDataGenerator = ImageDataGenerator(rescale=1/.255)
validationGenerator = validationDataGenerator.flow_from_directory(
    val,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

# define model

emotRecModel = Sequential()
emotRecModel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotRecModel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotRecModel.add(MaxPooling2D(pool_size=(2, 2)))
emotRecModel.add(Dropout(0.25))
emotRecModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotRecModel.add(MaxPooling2D(pool_size=(2, 2)))
emotRecModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotRecModel.add(MaxPooling2D(pool_size=(2, 2)))
emotRecModel.add(Dropout(0.25))
emotRecModel.add(Flatten())
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





