from tensorflow.keras import Model
from tensorflow.keras.layers import Input, ReLU, Concatenate, Conv2D,\
                                    BatchNormalization, AvgPool2D, GlobalAvgPool2D, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import softmax
from tensorflow.keras.backend import image_data_format


class DenseNetBuilder:

    def __init__(self, net_config):
        self.__NetConfig = net_config
        self.__visibleLayer = None
        self.__model = None

    def build(self):
        """
        this function responsible for aggregating all components of our dense network
        :return: DenseNet Model
        """
        self.__initialize()
        for i in range(1, self.__NetConfig.NumDenseBlocks):
            self.__dense_block(block_idx=i-1)
            self.__transition(block_idx=i-1)
            print(f"Block {i} Added...")
        # block_idx = -1
        self.__dense_block(block_idx=int(self.__NetConfig.NumDenseBlocks)-1)
        print(f"Block {self.__NetConfig.NumDenseBlocks} Added (Last Block)...")
        self.__output()
        print("Output Layer Added...")
        self.__model = Model(inputs=self.__visibleLayer, outputs=self.__model, name="DenseNetModel")
        print("Successful DensNet Build...")
        return self.__model

    def __dense_block(self, block_idx):
        """
        this function responsible for adding of Dense Block
        :param block_idx: index of the block will be built in the network(just for naming)
        """
        merge_tensor = [self.__model]
        for i in range(self.__NetConfig.NumLayers[block_idx]):
            """ both layers have the same number of filters = GrowthRate """
            """ bottleneck layer for regularization """
            self.__bn_relu_conv(kernel_size=1, filters=self.__NetConfig.GrowthRate,
                                name=f"Block_{block_idx+1}__Conv1X1_{i+1}")
            self.__bn_relu_conv(kernel_size=3, filters=self.__NetConfig.GrowthRate,
                                name=f"Block_{block_idx+1}__Conv3X3_{i+1}")

            # updating number of filters applied right now
            self.__NetConfig.NumberOfFilters += self.__NetConfig.GrowthRate

            merge_tensor.append(self.__model)
            self.__model = Concatenate()(merge_tensor)

        """ temp_tensor = self.__model
         self.__model = Concatenate()([merge_tensor, self.__model])
         merge_tensor.append(temp_tensor)"""

    def __transition(self, block_idx):
        """
        this function add transition layer between 2 blocks
        :param block_idx: index of the block before this transition(just for naming)
        """
        # this conv to reduce depth for regularization
        self.__bn_relu_conv(kernel_size=(1, 1), filters=self.__NetConfig.NumberOfFilters,
                            name=f"Block_{block_idx+1}__Trans_Conv")
        self.__model = AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                 name=f"Block_{block_idx+1}__Trans_AvgPool")(self.__model)

    def __output(self):
        """
        this function add last layer in our model which is fully connected layer
        """
        # image_data_format() => Returns the default image data format convention
        # a string, either 'channels_first' or 'channels_last'
        self.__model = BatchNormalization(name="Output-BN", momentum=self.__NetConfig.momentum_term,
                                          gamma_regularizer=l2(self.__NetConfig.weight_decay),
                                          beta_regularizer=l2(self.__NetConfig.weight_decay))(self.__model)
        self.__model = ReLU(name="Output-ReLU")(self.__model)
        self.__model = GlobalAvgPool2D(data_format=image_data_format())(self.__model)
        self.__model = Dense(units=self.__NetConfig.NumOfClasses, activation=softmax,
                             kernel_regularizer=l2(self.__NetConfig.weight_decay))(self.__model)

    def __bn_relu_conv(self, kernel_size, filters, name):
        """
        this function responsible for adding conv layer
        (in denseNet represented with composite layer: BN, ReLU then Conv)
        :param kernel_size: filter dimensions
        :param filters: number of filters (feature maps)
        :param name: for naming
        """
        def apply_bn():
            self.__model = BatchNormalization(name=f"{name}_BN", momentum=self.__NetConfig.momentum_term,
                                              gamma_regularizer=l2(self.__NetConfig.weight_decay),
                                              beta_regularizer=l2(self.__NetConfig.weight_decay))(self.__model)

        def apply_relu(): self.__model = ReLU(name=f"{name}_ReLU")(self.__model)

        def apply_conv(): self.__model = Conv2D(kernel_size=kernel_size, filters=filters, name=name,
                                                kernel_regularizer=l2(self.__NetConfig.weight_decay)
                                                , kernel_initializer="he_uniform", padding="same")(self.__model)
        apply_bn()
        apply_relu()
        apply_conv()

    def __initialize(self):
        """
        this function adding input later and initial convolution layer
        """
        # Adding Input Layer
        self.__visibleLayer = Input(shape=[self.__NetConfig.InputShape, self.__NetConfig.InputShape, 1],
                                    name='visible_layer')
        self.__model = Conv2D(filters=self.__NetConfig.NumberOfFilters, kernel_size=(7, 7), strides=2, padding="same",
                              kernel_regularizer=l2(self.__NetConfig.weight_decay), kernel_initializer="he_uniform",
                              name="initial_convolution")(self.__visibleLayer)

        # pooling layer decreases dimensions to half
        # self.__model = MaxPool2D(pool_size=3, strides=2, padding='same', name="initial_MaxPool")(self.__model)

        print("Initial Layers Added...")










