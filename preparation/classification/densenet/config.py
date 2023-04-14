import os
# 全局配置
# 数据集相关
ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))  # 当前文件所在路径
CLASS_BAD_DIR = 'diseased'  # 标注为有缺陷的图像所在路径
CLASS_GOOD_DIR = 'normal'  # 标注为无缺陷的图像所在路径
DATA_DIR = 'data'
TRAIN_DIR = 'train'  # 训练数据集所在路径
TEST_DIR = 'test'  # 测试数据集所在路径
TAR = 0.8  # TRAIN_ALL_RATIO, 训练数据集在总数据集中的占比

# 模型相关
BATCH_SIZE = 2
LR = 0.001  # 学习率
EPOCH = 100  # 训练轮数
LR_DECAY_STEP = 5 # 下降轮数

# 调试相关
LOG_DIR = './logs'

