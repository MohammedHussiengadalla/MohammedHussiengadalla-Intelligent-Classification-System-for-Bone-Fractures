import os
import numpy as np
import cv2
from sklearn.utils import shuffle

from netconfig import NetConfig


class DataLoader:
    def __init__(self, net_config):

        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.conf = net_config

    @staticmethod
    def get_label(img_name):
        if img_name.split('_')[-1] == "positive":
            return [1, 0]
        elif img_name.split('_')[-1] == "negative":
            return [0, 1]
        else:
            raise Exception("Error in label detection")

    def __get_train_data(self):
        training_set = []
        count = 0
        for root, dirs, files in os.walk(self.conf.RawTrainSetDir):
            for img in files:
                path = os.path.join(root, img)
                img_data = cv2.imread(path, 0)
                try:
                    img_data = cv2.resize(img_data, dsize=(self.conf.InputShape, self.conf.InputShape))
                except Exception:
                    print("Exception in resizing :", root)

                training_set.append([np.array(img_data), self.get_label(os.path.basename(root))])
                count += 1
                print("Images Loaded : ", count)
        training_set = shuffle(training_set, random_state=0)
        return np.array(training_set)

    def __get_valid_data(self):
        validation_set = []
        count = 0
        for root, dirs, files in os.walk(self.conf.RawValidSetDir):
            for img in files:
                path = os.path.join(root, img)
                img_data = cv2.imread(path, 0)
                try:
                    img_data = cv2.resize(img_data, dsize=(self.conf.InputShape, self.conf.InputShape))
                except Exception:
                    print("Exception in resizing :", root, img)
                validation_set.append([np.array(img_data), self.get_label(img_name=os.path.basename(root))])
                count += 1
                print("Images Loaded : ", count)
        validation_set = shuffle(validation_set, random_state=0)
        return validation_set

    def load(self):
        # Loading Train Set
        print("start data Loading...!")
        print("Loading Training Set...!")
        if os.path.exists(self.conf.NpyTrainSetDir):
            print("     => Loading From exist Numpy File...!")
            training_set = np.load(self.conf.NpyTrainSetDir, allow_pickle=True)
        else:
            print("     => Loading From Raw Data Source...!")
            training_set = self.__get_train_data()
            np.save(file=self.conf.NpyTrainSetDir, arr=training_set)

        self.x_train = np.array([i[0] for i in training_set])
        self.x_train = np.expand_dims(self.x_train, axis=3)
        self.y_train = np.array([i[1] for i in training_set])

        # Loading Validation Set
        print("Loading Validation Set...!")
        if os.path.exists(self.conf.NpyValidSetDir):
            print("     => Loading From exist Numpy File...!")
            validation_set = np.load(self.conf.NpyValidSetDir, allow_pickle=True)
        else:
            print("     => Loading From Raw Data Source...!")
            validation_set = self.__get_valid_data()
            np.save(file=self.conf.NpyValidSetDir, arr=validation_set)

        self.x_valid = np.array([i[0] for i in validation_set])
        self.x_valid = np.expand_dims(self.x_valid, axis=3)
        self.y_valid = np.array([i[1] for i in validation_set])

        print("End of Loading data..!")

    @staticmethod
    def display(images, images_name):
        lim = 10
        for i in range(lim):
            cv2.imshow(images_name, images[i])
            cv2.waitKey(0)


if __name__ == '__main__':
    conf = NetConfig()
    data = DataLoader(net_config=conf)
    data.load()
    print(data.y_train.shape)
    print(data.x_train.shape)
    print(data.x_valid.shape)
    print(data.y_valid.shape)
    print("y_train", data.y_train)
    data.display(data.x_train, "x_train")

    print("y_valid", data.y_valid)
    data.display(data.x_valid, "x_valid")


'''class DataLoader:

    def __init__(self, train_path=None, valid_path=None):
        self.__dim = (224, 224)
        self.conf.RawTrainSetDir = train_path
        self.conf.RawValidSetDir = valid_path
        self.__xdirs = []
        self.x_train = []
        self.y_train = []
        self.x_valid = []
        self.y_valid = []

    def __get_y_xdirs(self, path, y):
        """Loading Training Data"""
        self.__xdirs = []
        for root, dirs, files in os.walk(path):
            for example in files:
                self.__xdirs += [os.path.join(root, example)]
                y += [1 if root.split('_')[-1] == 'positive' else 0]
        np.asarray(y)

    def __get_x(self, x):
        for examples_dir in self.__xdirs:
            example = cv2.imread(examples_dir, 0)
            example = cv2.resize(example, self.__dim)
            x += [example]
        self.__normalize(x)

    def __normalize(self, x):
        x = np.asarray(x).astype('float32')
        mean = np.mean(x)
        std = np.std(x)
        x = (x - mean) / std

    def load_data(self):
        """Load Training Data"""
        self.__get_y_xdirs(self.conf.RawTrainSetDir, self.y_train)
        self.__get_x(self.x_train)
        self.__get_y_xdirs(self.conf.RawValidSetDir, self.y_valid)
        self.__get_x(self.x_valid)
        """
             """"""Load Validation Data""""""
             self.__get_y_xdirs(self.conf.RawValidSetDir)
             self.__get_x()
             x_valid = self.x
             y_valid = self.y"""



    @staticmethod
    def display(images, images_name):
        for image in images:
            cv2.imshow(images_name, image)
            cv2.waitKey(0)
'''
