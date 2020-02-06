import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def transform_y(train_y):
    type_list = []
    label_y = np.zeros((len(train_y), 6), dtype=int)
    labl = 0
    idx = 0
    for item in train_y:
        if np.size(type_list) == 0:
            type_list.append([item, labl])
            label_y[idx, labl] = 1
            labl += 1
        else:
            record = 0
            pos = 0
            for inst in type_list:
                if item == inst[0]:
                    label_y[idx, pos] = 1
                    record = 1
                    break
                pos += 1
            if record == 0:
                type_list.append([item, labl])
                label_y[idx, labl] = 1
                labl += 1
        idx += 1
    return label_y, type_list

train_x = np.load("train_data.npy")
train_y = np.load("train_label.npy")

train_x = train_x.reshape(train_x.shape[0], 64, 64, 1)
label_y, type_list = transform_y(train_y)

X_train, X_test, y_train, y_test = train_test_split(train_x, label_y, random_state=0)

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(64,64,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.1))

#model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(Dropout(0.1))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.1))

model.add(Flatten())

#model.add(Dense(16, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=16)