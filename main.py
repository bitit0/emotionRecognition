import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

emotion_key = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

raw_data_csv_file_name = "fer2013/fer2013.csv"
raw_data = pd.read_csv(raw_data_csv_file_name)



plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    x_pixels = np.array(raw_data["pixels"][i].split(" "), 'float32')
    x_pixels /= 255
    x_reshaped = x_pixels.reshape(48, 48)

    plt.imshow(x_reshaped, cmap="gray", interpolation="nearest")
    plt.xlabel(emotion_key[int(raw_data["emotion"][i])])
plt.show()