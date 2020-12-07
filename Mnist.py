from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.utils import to_categorical
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#load and process
def load_and_process(data_path):
    data = pd.read_csv(data_path)
    data = data.values
    np.random.shuffle(data)
    x = data[:,1:].reshape(-1,28,28,1)/255.0
    y = data[:,0].astype(np.int32)
    y = to_categorical(y, num_classes=len(set(y)))

    return x,y

#Data Path
train_data_path = "datasets/train.csv"
test_data_path = "datasets/test.csv"

x, y = load_and_process(train_data_path)
test = pd.read_csv(test_data_path)
test = x_test.to_numpy()
np.random.shuffle(test)
test = test[:,:].reshape(-1,28,28,1)/255.0

#Data Visualition
index = 11
vis = x.reshape(42000,28,28)
plt.imshow(vis[index,:,:])
plt.legend()
plt.axis("off")
plt.show()
print(np.argmax(y[index]))

#Model
numberOfClass = y.shape[1]

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (3,3), input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 128, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units = 256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(units = numberOfClass)) #output
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
                optimizer = "adam",
                metrics = ["accuracy"])

#Model Training
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33)

hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=2000)

model.save_weights("deneme.h5")

#Model Evaluation
print(hist.history.keys())
plt.plot(hist.history["loss"],label = "Train Loss")
plt.plot(hist.history["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"],label = "Train accuracy")
plt.plot(hist.history["val_accuracy"],label = "Validation accuracy")
plt.legend()
plt.show()

y_pred = model.predict_classes(test)
sub = pd.read_csv("datasets/sample_submission.csv")
sub["Label"] = y_pred
sub.to_csv("my_submission.csv", index = None)

#Saving History
import json

with open("deneme.json","w") as f:
    json.dump(hist.history, f)

#Loading History
import codecs

with codecs.open("deneme.json","r",encoding="utf-8") as f:
    h = json.loads(f.read())

print(hist.history.keys())
plt.plot(h["loss"],label = "Train Loss")
plt.plot(h["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["accuracy"],label = "Train accuracy")
plt.plot(h["val_accuracy"],label = "Validation accuracy")
plt.legend()
plt.show()