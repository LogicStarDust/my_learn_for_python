"""Trains a simple convnet on the MNIST dataset.
If implemented correctly, it is easy to get over 99.2% for testing accuracy
after a few epochs. And thre is still a lot of margin for parameter tuning,
e.g. different initializations, decreasing learning rates when the accuracy
stops increasing, or using model ensembling techniques, etc.
在MNIST数据集上训练一个简单的convnet。
如果正确执行，测试精度很容易超过99.2％
经过几个epochs。 而对于参数调整来说，仍然有很大的余地，
例如 不同的初始化，精度降低学习率
停止增加，或使用模型组合技术等。
"""

from __future__ import print_function

import os

import keras
from keras import backend as k
from keras import optimizers
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
# 图片输入大小
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("训练集数据维度:", x_train.shape, ",其标签维度:", y_train.shape)
# util.printImgFromMNIST(x_train, y_train, 1)

# 格式化数据。把训练集从（n,28,28）的形式变成（n,1,28,28）或者（n,28,28,1），其中1代表通道
# 原始数据代表有n个数据，一个数据28行28列，每一个元素存放一个像素值
# 转化数据代表有n个数据，一个数据38行38列，每一个又有一个数组，这里数组大小为1，放置一个像素值
if k.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# In order to speed up the convergence, we may normalize the input values
# so that they are in the range of (0, 1) for (-1, 1)
# Your code here.
# 为了快速收敛，我们可以标准化输入值。所以它们的范围从（0，1）到（-1，1）
x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
X_train = x_train.astype("float32") / 255.0
X_test = x_test.astype("float32") / 255.0

# Convert class vectors to binary class matrices, e.g. "1" ==> [0,1,0,0,0,0,0,0,0,0]
# 转换类向量成二进制矩阵，如"1" ==> [0,1,0,0,0,0,0,0,0,0]
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("格式化后训练集数据维度:", x_train.shape, ",其标签维度:", y_train.shape)
print(x_train.shape[0], '个训练数据')
print(x_test.shape[0], '个测试数据')

# 序贯模型是多个网络层的线性堆叠，可以通过.add()方法一个个的将layer加入模型
model = Sequential()

# Please build up a simple ConvNets by stacking a few conovlutioanl layers (kenel size with 3x3
# is a good choice, don't foget using non-linear activations for convolutional layers),
# max-pooling layers, dropout layers and dense/fully-connected layers.
# 请建立简单的ConvNets通过堆积几个卷积层（核心大小为3x3）是个好的选择，不要忘记使用非线性激活给卷积层、
# 最大池化层，dropout层和dense/fully-connected 层.
# Your code here.s
# 输入层有784个神经元
# 第一个隐层有512个神经元，激活函数为ReLu，Dropout比例为0.2
model.add(Dense(512, input_shape=(784,), activation="relu"))

# 第二个隐层有512个神经元，激活函数为ReLu，Dropout比例为0.2
model.add(Dense(512, activation="relu"))

# 输出层有10个神经元，激活函数为SoftMax，得到分类结果
model.add(Dense(10, activation="softmax"))

# complete the loss and optimizer
# Hints: use the cross-entropy loss, optimizer could be SGD or Adam, RMSProp, etc.
# Feel free to try different hyper-parameters.
# 计算损失函数和优化器。提示：使用交叉熵损失、优化器可以使用sgd或Adam, RMSProp。请随意尝试不同的超参数。
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=['accuracy'])

# Extra Points 1: use data augmentation in the progress of model training.
# Note that data augmentation is a practical technique for reducing overfitting.
# Hints: you may refer to https://keras.io/preprocessing/image/
# 使用数据扩增在这个模型训练过程。注意，数据增强是一种减少过度拟合的实用技术。
# Your code here
# x_train = data_augment(...)


# Extra Points 2: use K-Fold cross-validation for ensembling k models
# i.e. (1) split the whole training data into K folds;
#      (2) train K models based on different training data;
#      (3) when evaludating the testing data, averaging over K model predictions as final output.
# 使用交叉严重和模型聚合。
#       （1）分割训练数据成k层
#       （2）训练k个模型基于不同的训练集
#       （3）当评估测试数据时候，平均超过k个模型的预测为最终输出
# The codes may look like:
#   for i in range(K):
#       x_train, y_train = ...
#       model_i = train(x_train , y_train)

#
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
