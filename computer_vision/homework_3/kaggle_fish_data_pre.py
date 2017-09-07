"""
照片数据预处理
"""
import os

# 数据地址
data_dir = "D:/mnist/train"
all_data_dir="D:/mnist/all_train"
# 分类标签： 长鳍金枪鱼、大眼金枪鱼、鲯鳅、月鱼、鲨鱼、黄鳍金枪鱼、其他鱼、没有鱼
fishes = ["ALB", "BET", "DOL", "LAG", "SHARK", "YFT", "OTHER", "NoF"]
for fish in fishes:
    if fish in os.listdir(data_dir):
        print("发现数据：", fish)
    else:
        print("未发现数据：", fish)
    total_images = os.listdir(os.path.join(all_data_dir, fish))
