import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, Dropout, Add
from tensorflow.keras.activations import relu, swish, sigmoid
from tensorflow_addons.layers import AdaptiveAveragePooling2D, StochasticDepth
from math import ceil


def conv2d_bn(X_input, filters, kernel_size, strides, padding='same', activation=None,
              name=None, groups=1):
    """
    Implementation of a convolutional - batch norm - activation block
    
    :param X_input: input tensor
    :param filters: number of filters in the CONV layer
    :param kernel_size: shape of the CONV kernel
    :param s: stride to be used
    :param padding: padding approach to be used
    :param name: layers' name
    :return: output of the layer
    """

    # defining name basis
    conv_name_base = 'conv_'
    bn_name_base = 'bn_'

    X = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding=padding, name=conv_name_base + name, groups=groups,
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name=bn_name_base + name)(X)
    if activation is not None:
        X = Activation(activation)(X)

    return X


# ---------------------------------- Inception V4 ---------------------------------------------------

def Stem_block(X_input):
    """
    Stem block of Inception V4
    
    :param X_input: input tensor
    :return: output of the block
    """

    # First conv 
    X = conv2d_bn(X_input, filters=32, kernel_size=(3, 3), strides=(2, 2),
                  padding='valid', activation='relu', name='stem_1')
    # Second conv
    X = conv2d_bn(X, filters=32, kernel_size=(3, 3), strides=(1, 1),
                  padding='valid', activation='relu', name='stem_2')
    # Third conv
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1),
                  padding='same', activation='relu', name='stem_3')
    # First branch: max pooling
    branch_a = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='valid', name='stem_4a')(X)
    # Second branch: conv
    branch_b = conv2d_bn(X, filters=96, kernel_size=(3, 3),
                         strides=(2, 2), padding='valid', activation='relu',
                         name='stem_4b')
    # Concatenate (1) branch_a and branch_b along the channel axis
    X = tf.concat(values=[branch_a, branch_b], axis=3)
    # First branch: 2 convs
    branch_a = conv2d_bn(X, filters=64, kernel_size=(1, 1),
                         strides=(1, 1), padding='same', activation='relu',
                         name='stem_5a_1')
    branch_a = conv2d_bn(branch_a, filters=96, kernel_size=(3, 3),
                         strides=(1, 1), padding='valid', activation='relu',
                         name='stem_5a_2')
    # Second branch: 4 convs
    branch_b = conv2d_bn(X, filters=64, kernel_size=(1, 1),
                         strides=(1, 1), padding='same', activation='relu',
                         name='stem_5b_1')
    branch_b = conv2d_bn(branch_b, filters=64, kernel_size=(7, 1),
                         strides=(1, 1), padding='same', activation='relu',
                         name='stem_5b_2')
    branch_b = conv2d_bn(branch_b, filters=64, kernel_size=(1, 7),
                         strides=(1, 1), padding='same', activation='relu',
                         name='stem_5b_3')
    branch_b = conv2d_bn(branch_b, filters=96, kernel_size=(3, 3),
                         strides=(1, 1), padding='valid', activation='relu',
                         name='stem_5b_4')
    # Concatenate (2) branch_a and branch_b along the channel axis
    X = tf.concat(values=[branch_a, branch_b], axis=3)
    # First branch: conv
    branch_a = conv2d_bn(X, filters=192, kernel_size=(3, 3),
                         strides=(2, 2), padding='valid', activation='relu',
                         name='stem_6a')
    # Second branch: max pooling
    branch_b = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='valid', name='stem_6b')(X)
    # Concatenate (3) branch_a and branch_b along the channel axis
    X = tf.concat(values=[branch_a, branch_b], axis=3)

    return X


def A_block(X_input, n_of_block):
    """
  A block of Inception V4

  :param X_input: input tensor
  :param n_of_block: number of the block to add to the layers name
  :return: output of the block
  """
    # set general name for A block
    name = 'a_block' + n_of_block
    # First branch: Avg pooling and convolution
    branch_a = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                padding='same', name=name + '_a1')(X_input)
    branch_a = conv2d_bn(X_input, filters=96, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_a2')
    # Second branch: convolution
    branch_b = conv2d_bn(X_input, filters=96, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_b')
    # Third branch: 2 convs
    branch_c = conv2d_bn(X_input, filters=64, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c1')
    branch_c = conv2d_bn(branch_c, filters=96, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c2')
    # Fourth branch: 3 convs
    branch_d = conv2d_bn(X_input, filters=64, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d1')
    branch_d = conv2d_bn(branch_d, filters=96, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d2')
    branch_d = conv2d_bn(branch_d, filters=96, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d3')
    # Concatenate branches along the channel axes
    X = tf.concat(values=[branch_a, branch_b, branch_c, branch_d], axis=3)

    return X


def B_block(X_input, n_of_block):
    """
  B block of Inception V4

  :param X_input: input tensor
  :param n_of_block: number of the block to add to the layers name
  :return: output of the block
  """
    # set general name for B block
    name = 'b_block' + n_of_block
    # First branch: Avg pooling and convolution
    branch_a = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                padding='same', name=name + '_a1')(X_input)
    branch_a = conv2d_bn(branch_a, filters=128, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_a2')
    # Second branch: convolution
    branch_b = conv2d_bn(X_input, filters=384, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_b')
    # Third branch: 3 convs
    branch_c = conv2d_bn(X_input, filters=192, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c1')
    branch_c = conv2d_bn(branch_c, filters=224, kernel_size=(1, 7), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c2')
    branch_c = conv2d_bn(branch_c, filters=256, kernel_size=(7, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c3')
    # Fourth branch: 5 convs
    branch_d = conv2d_bn(X_input, filters=192, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d1')
    branch_d = conv2d_bn(branch_d, filters=192, kernel_size=(1, 7), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d2')
    branch_d = conv2d_bn(branch_d, filters=224, kernel_size=(7, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d3')
    branch_d = conv2d_bn(branch_d, filters=224, kernel_size=(1, 7), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d4')
    branch_d = conv2d_bn(branch_d, filters=256, kernel_size=(7, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d5')
    # Concatenate branches along the channel axes
    X = tf.concat(values=[branch_a, branch_b, branch_c, branch_d], axis=3)

    return X


def C_block(X_input, n_of_block):
    """
  C block of Inception V4

  :param X_input: input tensor
  :param n_of_block: number of the block to add to the layers name
  :return: output of the block
  """
    # set general name for C block
    name = 'c_block' + n_of_block
    # First branch: Avg pooling and convolution
    branch_a = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                padding='same', name=name + '_a1')(X_input)
    branch_a = conv2d_bn(branch_a, filters=256, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_a2')
    # Second branch: convolution
    branch_b = conv2d_bn(X_input, filters=256, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_b')
    # Third branch: 3 convs
    branch_c = conv2d_bn(X_input, filters=384, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c1')
    branch_ca = conv2d_bn(branch_c, filters=256, kernel_size=(1, 3), strides=(1, 1),
                          padding='same', activation='relu', name=name + '_c2a')
    branch_cb = conv2d_bn(branch_c, filters=256, kernel_size=(3, 1), strides=(1, 1),
                          padding='same', activation='relu', name=name + '_c2b')
    # Fourth branch: 5 convs
    branch_d = conv2d_bn(X_input, filters=384, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d1')
    branch_d = conv2d_bn(branch_d, filters=448, kernel_size=(1, 3), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d2')
    branch_d = conv2d_bn(branch_d, filters=512, kernel_size=(3, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_d3')
    branch_da = conv2d_bn(branch_d, filters=256, kernel_size=(3, 1), strides=(1, 1),
                          padding='same', activation='relu', name=name + '_d4a')
    branch_db = conv2d_bn(branch_d, filters=256, kernel_size=(1, 3), strides=(1, 1),
                          padding='same', activation='relu', name=name + '_d4b')
    # Concatenate branches along the channel axes
    X = tf.concat(values=[branch_a, branch_b, branch_ca, branch_cb, branch_da, branch_db], axis=3)

    return X


def red_A_block(X_input):
    """
  First reduction block of Inception V4

  :param X_input: input tensor
  :return: output of the block
  """

    # set general name for red block
    name = 'red_a_block'
    # First branch: max pool with stride 2
    branch_a = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='valid', name=name + '_a')(X_input)
    # Second branch: 3x3 convolution
    branch_b = conv2d_bn(X_input, filters=384, kernel_size=(3, 3), strides=(2, 2),
                         padding='valid', activation='relu', name=name + '_b')
    # Third branch: 3 convs
    branch_c = conv2d_bn(X_input, filters=192, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c1')
    branch_c = conv2d_bn(branch_c, filters=224, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c2')
    branch_c = conv2d_bn(branch_c, filters=256, kernel_size=(3, 3), strides=(2, 2),
                         padding='valid', activation='relu', name=name + '_c3')
    # Concatenate branches along the channel axes
    X = tf.concat(values=[branch_a, branch_b, branch_c], axis=3)

    return X


def red_B_block(X_input):
    """
  Second reduction block of Inception V4

  :param X_input: input tensor
  :return: output of the block
  """

    # set general name for red block
    name = 'red_b_block'
    # First branch: max pool with stride 2
    branch_a = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='valid', name=name + '_a')(X_input)
    # Second branch: 2 convs
    branch_b = conv2d_bn(X_input, filters=192, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_b1')
    branch_b = conv2d_bn(branch_b, filters=192, kernel_size=(3, 3), strides=(2, 2),
                         padding='valid', activation='relu', name=name + '_b2')
    # Third branch: 4 convs
    branch_c = conv2d_bn(X_input, filters=256, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c1')
    branch_c = conv2d_bn(branch_c, filters=256, kernel_size=(1, 7), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c2')
    branch_c = conv2d_bn(branch_c, filters=320, kernel_size=(7, 1), strides=(1, 1),
                         padding='same', activation='relu', name=name + '_c3')
    branch_c = conv2d_bn(branch_c, filters=320, kernel_size=(3, 3), strides=(2, 2),
                         padding='valid', activation='relu', name=name + '_c4')
    # Concatenate branches along the channel axes
    X = tf.concat(values=[branch_a, branch_b, branch_c], axis=3)

    return X


def Inception_v4(input_shape):
    """
  Inception v4 architecture

  :param input_shape: shape of input vector
  :return: Keras Model() element
  """

    X_input = Input(input_shape)

    X = Stem_block(X_input)

    for i in range(1, 5):  X = A_block(X, str(i))

    X = red_A_block(X)

    for i in range(1, 8): X = B_block(X, str(i))

    X = red_B_block(X)

    for i in range(1, 4): X = C_block(X, str(i))

    pooling_shape = X.get_shape()[1:3]
    X = AveragePooling2D(pooling_shape, name='last_avg_pool')(X)
    X = Flatten()(X)

    X = Dropout(rate=0.2)(X)

    X = Dense(3, activation='linear', name='last_layer')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='Inception_v4')

    return model

    # ---------------------------------- ResNet 50 ---------------------------------------------------


def res_identity(X_input, filter_size, name):
    """
  Identity block of ResNet50

  :param X_input: input tensor
  :param filter_size: list of 2 sizes for the different steps
  :param name: name associated with the block
  :return: output tensor
  """
    X_skip = X_input
    filter1, filter2 = filter_size
    id_name = lambda n: 'id_' + n + '_' + name

    # Standard path
    X = conv2d_bn(X_input, filters=filter1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                  activation='relu', name=id_name('a'))
    X = conv2d_bn(X, filters=filter1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                  activation='relu', name=id_name('b'))
    X = conv2d_bn(X, filters=filter2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                  activation=None, name=id_name('c'))

    # Add the skip connection
    X = Add()([X, X_skip])
    X = Activation(relu)(X)

    return X


def res_conv(X_input, filter_size, name, stride=2):
    """
  Convolutional block of ResNet50 - similar to identity but a convolution is applied to the skip-connection tensor

  :param X_input: input tensor
  :param filter_size: list of 2 sizes for the different steps
  :param name: name associated with the block
  :param stride: int value, stride of first and 'skip' convolutional blocks
  :return: output tensor
  """
    X_skip = X_input
    filter1, filter2 = filter_size
    id_name = lambda n: 'cv_' + n + '_' + name
    s = stride

    # Standard path
    X = conv2d_bn(X_input, filters=filter1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                  activation='relu', name=id_name('a'))
    X = conv2d_bn(X, filters=filter1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                  activation='relu', name=id_name('b'))
    X = conv2d_bn(X, filters=filter2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                  activation=None, name=id_name('c'))

    # Skip-connection path
    X_skip = conv2d_bn(X_skip, filters=filter2, kernel_size=(1, 1), strides=(s, s),
                       padding='valid', activation=None, name=id_name('skip'))

    # Add the skip connection
    X = Add()([X, X_skip])
    X = Activation(relu)(X)

    return X


def ResNet50(input_shape):
    """
  ResNet50 architecture

  :param input_shape: shape of input vector
  :return: Keras Model() element
  """
    # Create input
    X_input = Input(input_shape)

    # Stage 0
    X = conv2d_bn(X_input, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid',
                  activation='relu', name='stage0')
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 1
    X = res_conv(X, (64, 256), 'stage1a', stride=1)
    X = res_identity(X, (64, 256), 'stage1b')
    X = res_identity(X, (64, 256), 'stage1c')

    # Stage 2
    X = res_conv(X, (128, 512), 'stage2a')
    X = res_identity(X, (128, 512), 'stage2b')
    X = res_identity(X, (128, 512), 'stage2c')
    X = res_identity(X, (128, 512), 'stage2d')

    # Stage 3
    X = res_conv(X, (256, 1024), 'stage3a')
    X = res_identity(X, (256, 1024), 'stage3b')
    X = res_identity(X, (256, 1024), 'stage3c')
    X = res_identity(X, (256, 1024), 'stage3d')
    X = res_identity(X, (256, 1024), 'stage3e')
    X = res_identity(X, (256, 1024), 'stage3f')

    # Stage 4
    X = res_conv(X, (512, 2048), 'stage4a')
    X = res_identity(X, (512, 2048), 'stage4b')
    X = res_identity(X, (512, 2048), 'stage4c')

    # Final stage
    X = AveragePooling2D((2, 2), padding='same')(X)
    X = Flatten()(X)
    X = Dense(3, activation='linear', name='last_layer')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


# ---------------------------------- Custom CNN ---------------------------------------------------

def CNN(input_shape, filters, compression_depth=3, verbose=False):
    """
  Customized CNN architecture with compression of input image shape

  :param input_shape: shape of input vector
  :param filters: list of filters to use to build the model
  :param compression_depth: number of layers where pictures' size is halved
  :return: Keras Model() element
  """
    # Function to print layer shape
    if verbose: ps = lambda X, name: print('Step {}: {}'.format(name, X))

    # Create input
    X_input = Input(input_shape)
    if verbose: ps(X_input, 'in')
    assert compression_depth <= len(filters), 'compression depth is greater than number of layers'
    cd = compression_depth - 1

    # Build layers according to filters
    for i, n_filters in enumerate(filters):
        if i == 0:
            X = conv2d_bn(X_input, n_filters, kernel_size=(4, 4), strides=(2, 2), padding='same',
                          activation='relu', name='l{}'.format(i))
        elif 1 <= i <= cd:
            X = conv2d_bn(X, n_filters, kernel_size=(4, 4), strides=(2, 2), padding='same',
                          activation='relu', name='l{}'.format(i))
        else:
            X = conv2d_bn(X, n_filters, kernel_size=(4, 4), strides=(1, 1), padding='same',
                          activation='relu', name='l{}'.format(i))
        if verbose: ps(X, i)

    pooling_shape = X.get_shape()[1:3]
    X = AveragePooling2D(pooling_shape, padding='same')(X)
    X = Flatten()(X)
    X = Dropout(0.2)(X)
    X = Dense(3, activation='linear', name='last_layer')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='custom_CNN')

    return model


# ---------------------------------- EfficientNet ---------------------------------------------------

def SqueezeExcitations(X_input, reduced_dimension, name=None):
    """
    SqueezeExcitation block to build a EfficientNet architecture

    :param X_input: input tensor
    :param reduced_dimension: dimention to wich reduce the input to generate the Squeeze-Excitation coefficients
    :param name: main name of the layer
    :return: tensor of same shape as input
    """
    input_shape = X_input.get_shape().as_list()[-1]

    sqex = AdaptiveAveragePooling2D((1, 1))(X_input)
    sqex = Conv2D(filters=reduced_dimension, kernel_size=1,
                  name='squeeze_' + name,
                  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(sqex)
    sqex = Activation(swish)(sqex)
    sqex = Conv2D(input_shape, kernel_size=1,
                  name='excite_' + name,
                  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(sqex)
    sqex = Activation(sigmoid)(sqex)

    return X_input * sqex


def InversedResidualBlock(X_input, filters, kernel_size, strides, padding, expand_ratio,
                          reduction=4, survival_prob=0.8, name=None):
    """
    Implementation of a inversed Residual block as in MobileNet

    :param X_input: input tensor
    :param filters: number of filters in the CONV layer
    :param kernel_size: shape of the CONV kernel
    :param strides: strides to be used
    :param padding: padding approach to be used
    :param expand_ratio: ratio of expanction in first convolution
    :param reduction: fatcor for which the input shape is reduced
    :param survival_prob: survival probability of the StochasticDepth layer
    :param name: layers' name
    :return: output of the layer
    """

    input_shape = X_input.get_shape().as_list()[-1]
    hidden_dim = input_shape * expand_ratio
    reduced_dim = input_shape // reduction
    use_residuals = input_shape == filters and strides == 1

    if expand_ratio != 1:
        X = conv2d_bn(X_input, filters=hidden_dim, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', activation='swish', name='expand_' + name)
    else:
        X = X_input

    X = conv2d_bn(X, filters=hidden_dim, kernel_size=kernel_size, strides=strides, padding=padding,
                  activation='swish', groups=hidden_dim, name='a_' + name)
    X = SqueezeExcitations(X, reduced_dimension=reduced_dim, name='b_' + name)
    X = conv2d_bn(X, filters=filters, kernel_size=(1, 1), strides=(1, 1), activation=None,
                  name='c_' + name)

    if use_residuals:
        X = StochasticDepth(survival_probability=survival_prob)([X, X_input])

    return X


def EfficientNet(input_shape, version, n_class=3):
    """
    :param input_shape: shape of input vector
    :param version: string containing the version of the EfficientNet architecture, e.g. 'b0', 'b1', ..., 'b7'
    :param n_class: number of different classes to predict
    :return: Keras Model() element
    """
    base_model = [  # expand_ratio, filters, repeats, stride, kernel_size
        [1, 16, 1, 1, 3],
        [6, 24, 2, 2, 3],
        [6, 40, 2, 2, 5],
        [6, 80, 3, 2, 3],
        [6, 112, 3, 1, 5],
        [6, 192, 4, 2, 5],
        [6, 320, 1, 1, 3]]

    X_input = Input(input_shape)

    def get_factors(version, alpha=1.2, beta=1.1):
        phi_values = {  # (phi_value, resolution, drop_rate).
            "b0": (0, 224, 0.2),
            "b1": (0.5, 240, 0.2),
            "b2": (1, 260, 0.3),
            "b3": (2, 300, 0.3),
            "b4": (3, 380, 0.4),
            "b5": (4, 456, 0.4),
            "b6": (5, 528, 0.5),
            "b7": (6, 600, 0.5)}

        phi, res, drop_rate = phi_values[version]
        depth_ratio = alpha ** phi
        width_ratio = beta ** phi
        return depth_ratio, width_ratio, drop_rate

    depth, width, drop_rate = get_factors(version)
    last_channel = ceil(1280 * width)

    # First layer
    X = conv2d_bn(X_input, filters=int(32 * width), kernel_size=(3, 3), strides=2, padding='same',
                  activation='swish', name='first')
    count_layer = 0
    for expand_ratio, filters, repeats, stride, kernel_size in base_model:
        count_layer += 1
        out_filt = int(4 * ceil((filters * width / 4)))
        layers_repeat = int(ceil(repeats * depth))

        for layer in range(layers_repeat):
            X = InversedResidualBlock(X, filters=out_filt, kernel_size=(kernel_size, kernel_size),
                                      strides=(stride if layer == 0 else 1),
                                      expand_ratio=expand_ratio,
                                      padding='same', name='{}.{}'.format(count_layer, layer))

    X = conv2d_bn(X, filters=last_channel, kernel_size=(1, 1), strides=(1, 1), padding='same',
                  name='last_conv_')

    # Last layers
    X = AdaptiveAveragePooling2D((1, 1))(X)
    X = Flatten()(X)
    X = Dropout(rate=0.2)(X)
    X = Dense(n_class, activation='linear', name='last_layer')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='EfficientNet_' + version)

    return model
