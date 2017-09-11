from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# 模型训练参数
learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_train_samples = 3019
nbr_validation_samples = 758
nbr_epochs = 25
batch_size = 32

# 训练和验证数据路径
train_data_dir = 'E:/fish/root_train'
val_data_dir = 'E:/fish/val_train'

# 鱼类别
FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

# InceptionV3_notop？
print('加载 InceptionV3 权重 ...')
InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                                input_tensor=None, input_shape=(299, 299, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

# 增加平均池化层和softmax输出层
print("增加平均池化层和softmax输出层.....")
output = InceptionV3_notop.get_layer(index=-1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(8, activation='softmax', name='predictions')(output)

# 模型定义
InceptionV3_model = Model(InceptionV3_notop.input, output)
InceptionV3_model.summary()

# 优化器和学习效率定义
optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)

# 模型创建
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 模型检查点和权重文件路径
best_model_file = "./weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)

# 数据扩增的参数，依次为：
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

# 验证集的数据扩展，只使用rescale
# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)

# 训练数据数据扩增
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
    # save_prefix = 'aug',
    classes=FishNames,
    class_mode='categorical')
# 验证数据数据扩增
validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
    # save_prefix = 'aug',
    classes=FishNames,
    class_mode='categorical')

# 训练模型
InceptionV3_model.fit_generator(
    train_generator,
    samples_per_epoch=nbr_train_samples,
    nb_epoch=nbr_epochs,
    validation_data=validation_generator,
    nb_val_samples=nbr_validation_samples,
    callbacks=[best_model])
