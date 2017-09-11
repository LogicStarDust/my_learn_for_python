from keras.models import Model
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, AveragePooling2D

# 图片输入大小
learn_rate = 1e-4
img_rows, img_cols = 1280, 720

# 输入数据路径
train_data_dir = ""
validation_data_dir = ""

# InceptionV3
# include_top：是否保留顶层的全连接网络
# weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
# input_tensor：可填入Keras tensor作为模型的图像输出tensor
# input_shape：可选，仅当include_top=False有效，应为长为3的tuple，指明输入图片的shape，
#   图片的宽高必须大于197，如(200,200,3)
print("InceptionV3 声明")
inceptionV3 = InceptionV3(include_top=False,
                          weights="imagenet",
                          input_tensor=None,
                          input_shape=(299, 299, 3))

# 获取inceptionV3的输出层
inceptionV3_output = inceptionV3.get_layer(index=1).output
# 建立平均池化层
inceptionV3_output = AveragePooling2D((8, 8), strides=(8, 8), name="avg_pool")(inceptionV3_output)
# flatten
inceptionV3_output = Flatten(name="flatten")(inceptionV3_output)
# 全链接层，输出为8分类，进行softmax激活
inceptionV3_output = Dense(8, activation="softmax", name="prediction")(inceptionV3_output)

# 通过前面建立的inceptionV3建立模型
inceptionV3_model = Model(inputs=inceptionV3.input, outputs=inceptionV3_output)
# inceptionV3_model.summary()

# 声明梯度下降优化器,
# lr:学习率
# momentum:大或等于0的浮点数，动量参数
# decay:大或等于0的浮点数，每次更新后的学习率衰减值
# nesterov: 布尔值，确定是否使用Nesterov动量
optimozer = SGD(lr=learn_rate, momentum=0.9, decay=0.0, nesterov=True)
