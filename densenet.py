import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from densenet_builder import DenseNetBuilder
from data_loader import *


class DenseNet:
    def __init__(self, net_config):
        self.conf = net_config
        self.net_builder = DenseNetBuilder(net_config)
        self.__model = self.net_builder.build()

    '''def train(self, x, y):
        if os.path.exists(self.conf.saved_model_dir):
            print("Loading Saved Model...!")
            self.__model = tf.keras.models.load_model(self.conf.saved_model_dir)
            print("End of Loading Saved Model...!")

        else:
            self.__compile()
            print("Start Training...")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.conf.log_dir, histogram_freq=1)
            self.__model.fit(x, y, batch_size=self.conf.batch_size, epochs=self.conf.epochs,
                             validation_split=self.conf.validation_split, callbacks=[tensorboard_callback])
            print("End of Training...")
            print("Start Saving Trained Model...")
            self.__model.save(self.conf.saved_model_dir)
            print("Model Saved...")
            '''

    def train(self, x, y, initial_epoch=None):
        # initial_epoch => the number of saved models
        init_epoch = 0
        if initial_epoch is None:
            print("is None!!")
            init_epoch = 0
            self.__compile()

        else:
            init_epoch = initial_epoch
            self.__model = tf.keras.models.load_model(self.conf.saved_model_dir + "_epoch:{}.h5".format(init_epoch))
            print("Model {} is loaded".format(init_epoch))

        print("Start Training...")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.conf.log_dir, histogram_freq=1)
        for i in range(init_epoch, self.conf.epochs):
            print('epoch {}: \n'.format(i + 1))
            self.__model.fit(x, y, batch_size=self.conf.batch_size, epochs=1 + i, initial_epoch=0 + i,
                             validation_split=self.conf.validation_split, callbacks=[tensorboard_callback])
            self.__model.save(self.conf.saved_model_dir + "_epoch:{}.h5".format(i + 1))
            print("Model of epoch {} Saved...".format(i + 1))

        print("End of Training...")

    def test(self, x, y, with_epochs):
        self.__model = tf.keras.models.load_model(self.conf.saved_model_dir + "_epoch:{}.h5".format(with_epochs))
        test_scores = self.__model.evaluate(x, y, verbose=2)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])

    def predict(self, img):
        return self.__model.predict(img)

    def __compile(self):
        print("Start Compiling...")
        self.__model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=self.conf.Eta),
                             metrics=['accuracy'])
        print("Successful Compiling...")

    def load_model_by_path(self, path):
        self.__model = tf.keras.models.load_model(path)

    def summary(self):
        print(self.__model.summary())

    def plot(self):
        print("Start Plotting...")
        plot_model(model=self.__model, show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    NetConf = NetConfig()
    Data = DataLoader(NetConf)
    Data.load()
    # cv2.imshow("images_name", Data.x_valid[0])
    # cv2.waitKey(0)
    print("x_train shape : ", np.shape(Data.x_train))
    print("y_train shape : ", np.shape(Data.y_train))
    print("x_test shape : ", np.shape(Data.x_valid))
    print("y_test shape : ", np.shape(Data.y_valid))
    # print(tf.keras.backend.image_data_format())
    dense_model = DenseNet(NetConf)

    dense_model.summary()
    # dense_model.plot()
    dense_model.train(Data.x_train, Data.y_train, 7)
    '''for i in range(1, 45):
      print("Test Moedel {}".format(i))
      dense_model.test(Data.x_valid, Data.y_valid, i)'''
    # print("actual value = ", dense_model.predict(Data.x_valid[0]))
    # print("target value = ", Data.y_valid[0])
