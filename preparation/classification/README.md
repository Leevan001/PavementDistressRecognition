## 基于DenseNet121模型的病害二分类

使用CFD数据集进行测试，对病害图像和正常图像进行分类，从而掌握加载预训练模型进行简单迁移学习的过程。

### 环境说明

```
torch                        1.13.0
torchvision                  0.14.0
```

### 文件说明

```
│  README.md
│
└─densenet
    │  config.py			# 配置文件
    │  Dataset.py			# 数据集加载
    │  Model.py				# DenseNet121预训练模型
    │  test.py				# 测试代码（使用模型对测试图片进行标注）
    │  train.py				# 模型训练代码（在预训练模型的基础上迁移）
    │
    ├─data					# 数据集（igonre, 请自行添加CFD数据集）
    │  ├─test				# 测试数据
    │  │  ├─diseased		# 有病害的数据图片文件夹
    │  │  │      207.jpg
    │  │  │      *.jpg
    │  │  │
    │  │  └─normal			# 没有病害的数据图片文件夹
    │  │          096.jpg
    │  │          *.jpg
    │  │
    │  └─train				# 训练数据
    │      ├─diseased
    │      │      001.jpg
    │      │      *.jpg
    │      │
    │      └─normal
    │              001.jpg
    │              *.jpg
    │
    └─results				# 训练结果（ignore）
       │  12-07_19-46001.pth
       │  12-07_19-46002.pth
       │
       └─12-07_19-46
               loss_curve.png
```

