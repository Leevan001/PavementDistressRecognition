# %%
import torch
import torch.nn as nn
from Dataset import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from config import *
from datetime import datetime
import torchvision.models as models
from PIL import Image, ImageFont, ImageDraw
from Model import DenseNetModel
import time

# %%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# 从tensor恢复原图
def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img

# 创建输出文件夹路径out_dir
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
out_dir = os.path.join(ROOT_DIR, "results", "out_" + time_str)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 构建MyDataset实例
test_data = WovenBagDataset(data_dir=os.path.join(DATA_DIR, TEST_DIR), mode="test")
# 构建DataLoder
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

# ==================构建模型
mynet = DenseNetModel
mynet.to(device)
mynet.load_state_dict(torch.load("./results/12-07_22-40011.pth", map_location=device))
mynet.eval()

# ==================损失函数
criterion = nn.CrossEntropyLoss()

correct_test = 0.
total_test = 0.
loss_test = 0.
tp = 0.
fp = 0.
tn = 0.
fn = 0.

with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        print("inputs size:", len(inputs))
        start = time.time()
        outputs = mynet(inputs)
        loss = criterion(outputs, labels)
        end = time.time()
        print(f"Using time: {end-start:.5f}")
        _, predicted = torch.max(outputs.data, 1)
        # 遍历每张预测图 并保存结果
        for j, img in enumerate(inputs):
            if predicted[j] == 0:
                if labels[j] == 0:
                    tn += 1
                else:
                    fn += 1
            else:
                if labels[j] == 1:
                    tp += 1
                else:
                    fp += 1
            img_pred = "normal" if predicted[j] == 1 else "diseased"
            img_label = "normal" if labels[j] == 1 else "diseased"
            img_rgb = transform_convert(img, transform_)
            img_draw = ImageDraw.Draw(img_rgb)
            ttf = ImageFont.truetype("DejaVuSans.ttf", 30)
            img_draw.text((50, 50), "predict:{}\nlabel:{}".format(img_pred, img_label), fill=(255, 63, 51), font=ttf)
            img_rgb.save(out_dir + "/{:0>3}_{}.jpg".format(i, j))

        total_test += labels.size(0)
        correct_test += (predicted == labels).squeeze().cpu().sum().numpy()
        loss_test += loss.item()
        print("batch:[{:0>2}/{:0>2}] is done!".format(i + 1, len(test_loader)))

# %%
loss_test_mean = loss_test / len(test_loader)
test_acc = correct_test / total_test
recall = tn/(tn + fp)
precision = tn/(tn + fn)
print(f"|Test is done!|test_acc:{test_acc:.4f}|loss_test:{loss_test_mean:.4f}|Recall:{recall:.4f}|Precision:{precision:.4f}")


# %%
