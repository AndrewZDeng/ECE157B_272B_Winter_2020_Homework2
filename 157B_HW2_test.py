import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix

def transform2label(label_y, type_list):
    category_y = []

    for ohcode in label_y:
        pos = np.argmax(ohcode)
        category_y.append(type_list[pos,0])

    return category_y

model = load_model("my_model.h5")

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
type_list = np.load("type_list.npy")

category_train = transform2label(y_train, type_list)
category_val = transform2label(y_val, type_list)

y_train_prd = model.predict(X_train)
category_train_prd = transform2label(y_train_prd, type_list)

y_val_prd = model.predict(X_val)
category_val_prd = transform2label(y_val_prd, type_list)

confus_mtx_train = confusion_matrix(category_train, category_train_prd,
                                    labels=['pass', 'total_loss', 'deform', 'nodule', 'edge', 'crack'])
print("Confusion matrix of training set:\n{}".format(confus_mtx_train))

confus_mtx_val = confusion_matrix(category_val, category_val_prd,
                                  labels=['pass', 'total_loss', 'deform', 'nodule', 'edge', 'crack'])
print("Confusion matrix of validation set:\n{}".format(confus_mtx_val))

test_x = np.load("test_data.npy")
X_test = test_x.reshape(test_x.shape[0], 64, 64, 1)
y_test_prd = model.predict(X_test)
category_test_prd = transform2label(y_test_prd, type_list)

print(category_test_prd)
np.save("test_label.npy", category_test_prd)