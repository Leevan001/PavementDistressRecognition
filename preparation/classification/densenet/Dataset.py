# 自带库
import os
import random
from PIL import Image
# 需要安装
from torch.utils.data import Dataset
from torchvision import transforms

# 图像预处理
transform_ = transforms.Compose([
    transforms.ToTensor()
])

class WovenBagDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.8, rng_seed=620, transform=transform_):
        self.mode = mode
        self.data_dir = data_dir
        self.rng_seed = rng_seed  # 随机种子，保证每一次随机的顺序一致，避免训练集与测试集有交叉
        self.split_n = split_n
        self.data_info = self._get_img_info()  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img_rgb = Image.open(path_img)
        img = img_rgb
        if self.transform is not None:
            img = self.transform(img)
        return img, label  # 返回预处理后的(图片, 标签)

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def _get_img_info(self):
         # flaw=0, perfect=1
        good_set = [(os.path.join(self.data_dir, 'normal', x), 1) for x in os.listdir(os.path.join(self.data_dir, 'normal')) if not os.path.isdir(os.path.join(self.data_dir, 'normal', x))]
        bad_set = [(os.path.join(self.data_dir, 'diseased', x), 0) for x in os.listdir(os.path.join(self.data_dir, 'diseased')) if not os.path.isdir(os.path.join(self.data_dir, 'diseased', x))]
        all_set = good_set + bad_set
        random.seed(self.rng_seed)      
        random.shuffle(all_set)  # 打乱图片
        split_idx = int(len(all_set) * self.split_n)  # 训练集:验证集 = split_n : 1-split_n
        # 根据mode选择数据集
        if self.mode == "train":
            return all_set[:split_idx]
        elif self.mode == "valid":
            return all_set[split_idx:]
        elif self.mode == "test":
            return all_set
        else:
            raise Exception("self.mode can not be recognized, only support 'train', 'valid', 'test'. ")

