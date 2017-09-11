import os
import numpy as np
import shutil

np.random.seed(2016)

root_train = 'E:/fish/root_train'
root_val = 'E:/fish/val_train'

root_total = 'E:/fish/total_train'

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.8

for fish in FishNames:
    # 如果这种鱼不在val_train中，则创建
    if fish not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, fish))

    # 全部这种鱼的全部图片
    total_images = os.listdir(os.path.join(root_total, fish))

    # 根据比例获取训练数据的个数
    nbr_train = int(len(total_images) * split_proportion)

    # 打乱图片顺序
    np.random.shuffle(total_images)

    # 切取训练图片数据
    train_images = total_images[:nbr_train]

    # 验证数据
    val_images = total_images[nbr_train:]

    # 把切取的训练图片复制到目标文件夹
    for img in train_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_train, fish, img)
        shutil.copy(source, target)
        nbr_train_samples += 1

    # 如果没有建立root_val文件夹
    if fish not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, fish))

    # 验证的图片复制到root_val
    for img in val_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_val, fish, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))
