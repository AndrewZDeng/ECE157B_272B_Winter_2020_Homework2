import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, AveragePooling2D

from sklearn.model_selection import train_test_split

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

X_train, X_val, y_train, y_val = train_test_split(train_x, label_y, random_state=0, train_size=0.9)

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(4,4), strides=(1,1), padding='valid', input_shape=(64,64,1), activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

# model.add(Conv2D(filters=6, kernel_size=(2,2), strides=(1,1), padding='valid', activation='relu'))
# model.add(AveragePooling2D(pool_size=(2,2)))
# model.add(Dropout(0.1))

model.add(Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(AveragePooling2D(pool_size=(4,4)))
model.add(Dropout(0.1))

model.add(Flatten())

# model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=20)

model.save("my_model.h5")

np.save("type_list.npy", type_list)
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)