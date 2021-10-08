import datetime


class NetConfig:

    def __init__(self):
        self.InputShape = 224
        self.InputDType = "float32"
        self.NumDenseBlocks = 3
        self.NumLayers = [6, 12, 32]
        self.NumOfClasses = 2
        self.NumberOfFilters = 16
        self.GrowthRate = 12
        self.Eta = 0.001
        self.batch_size = 50
        self.validation_split = 0.2
        self.epochs = 100
        self.weight_decay = 0.0001
        self.momentum_term = 0.9
        self.RawTrainSetDir = r'raw_data/training_set'
        self.RawValidSetDir = r'raw_data/validation_set'
        self.NpyTrainSetDir = r"U:\GP\MURA-v1.1_npy\XR_HUMERUS_npy\XR_HUMERUS_training_set.npy"
        self.NpyValidSetDir = r"U:\GP\MURA-v1.1_npy\XR_HUMERUS_npy\XR_HUMERUS_validation_set.npy"
        self.log_dir = "logs/fit/HUMERUS_model"  # + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.saved_model_dir = "saved_model/HUMERUS_model"

        """
           :var InputShape: int -- dimension of input images.
           :var InputDType: string -- data type of pixel values.
           :var NumDenseBlocks: int -- number of dense blocks to be constructed in the network.
           :var NumLayers: int -- decide number of conv layers in each dense block.
           :var NumOfClasses: int -- number of classes to which model assign each example.
           :var NumberOfFilters: int -- number of filters in initial conv,
                number of filters in conv within transition = L*GrowthRate + NumberOfFilters.
           :var GrowthRate: float -- decide number of filters in each convolution layer.
           :var Eta: float -- learning rate
           :var batch_size: int -- mini batch size.
           :var validation_split: float -- portion of training data to be used in validation.
           :var epochs: int -- number of epochs while training.
           :var weight_decay (lambda): float -- (regularization strength)every coefficient in the weight matrix
                of the layer will add alpha*weight_coefficient_value**2 to the total loss of the network to perform
                l2 norm regularization.
           :var momentum_term (alpha): float -- momentum term used for regularization in BN layers.
           :var RawTrainSetDir: string -- directory of training data in its raw form.
           :var NpyTrainSetDir: string -- directory of testing data in its raw form.
           :var RawTrainSetDir: string -- directory of training data as .npy file.
           :var NpyValidSetDir: string -- directory of testing data as .npy file.
           :var log_dir: string -- directory to store training/validation log.
           :var saved_model_dir: string -- directory for saving models after training.
        """


class DensNet121(NetConfig):
    def __init__(self):
        super().__init__()
        self.NumDenseBlocks = 4
        self.NumLayers = [6, 12, 24, 16]
        self.L = 121


class DensNet169:
    def __init__(self):
        super().__init__()
        self.NumDenseBlocks = 4
        self.NumLayers = [6, 12, 32, 32]
        self.L = 169


class DensNet201:
    def __init__(self):
        super().__init__()
        self.NumDenseBlocks = 4
        self.NumLayers = [6, 12, 48, 32]
        self.L = 169


class DensNet264:
    def __init__(self):
        super().__init__()
        self.NumDenseBlocks = 4
        self.NumLayers = [6, 12, 64, 48]
        self.L = 169
