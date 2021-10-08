import numpy as np
import os


def update(data_set_as_npy, new_name):
    """
    this function convert from 'One Hot Encoding' to 'Label Encoding'
    which required in case we use 'SparseCategoricalCrossentropy' as lose function.

    :param data_set_as_npy: directory of numpy file with 'One Hot Encoding'.
    :param new_name: the name of generated numpy file with 'Label Encoding'.
    :return: no return, but it save the new file at the same directory of old file.
    """
    data_set = np.load(data_set_as_npy, allow_pickle=True)

    print("the shape of old data set", data_set.shape)
    print("shape of X in old data set", data_set[0][0].shape)
    print("shape of y in old data set", data_set[0][1])

    x = np.array([i[0] for i in data_set])
    y = np.array([i[1] for i in data_set])

    # =============================================
    #                 reshaping Y
    # =============================================

    print("shape of Y", y.shape)
    for i in range(10):
        print(y[i])

    y_new = np.array([])
    for i in range(y.shape[0]):
        if np.array_equal(y[i], [1, 0]):
            y_new = np.append(y_new, 1)
        elif np.array_equal(y[i], [0, 1]):
            y_new = np.append(y_new, 0)
        else:
            raise Exception("Error in Original Content.!!")
    y_new = y_new.astype("int")

    print("shape of Y after reshaping", y_new.shape)
    for i in range(10):
        print(y_new[i])

    # =============================================
    #       getting new Data Set
    # =============================================
    data_set_new = []
    for i in range(x.shape[0]):
        data_set_new.append([np.array(x[i]), y_new[i]])

    data_set_new = np.array(data_set_new)
    print("shape of new Data set", data_set_new.shape)
    print("shape of X in new Data set", data_set_new[0][0].shape)
    print("shape of y in new Data set", data_set_new[0][1])

    np.save(os.path.dirname(data_set_as_npy)+"/"+new_name, data_set_new)


update(data_set_as_npy="U:\GP\MURA-v1.1_npy\XR_HUMERUS_npy\XR_HUMERUS_training_set.npy",
       new_name="XR_HUMERUS_training_set_new")

