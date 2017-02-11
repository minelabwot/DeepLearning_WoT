### 基本介绍：

* 目标

本项目主要是基于深度学习模型完成从物联网时间序列源数据中提取高级行为模式信息。基于该数据集OPPORTUNITY而言，具体则是针对人们的行为模式根据源数据实现端到端的对人们下一时刻行为的预测。

* 创新点

与以往不同的是，为了使得程序可以更加智能的理解人们行为与意图之间的关系从而做出更加准确的预测，我们借鉴自然语言处理中对文本预测的成功经验，并综合动作识别模型DeepConvLSTM与视觉识别模型LRCN中的优点，建立了DeepConvNet深度网络模型。

***

### 项目架构：

* 模型示意图：

![](model1.png)

* 模型介绍：

因为动作的连续性与耗时性，我们通过一个合适长度window_length的时间窗对其通过CNN+LSTM的混合网络模型DeepConvLSTM提取特征，即得到每个动作的特征向量表示（动作识别任务也只需要做这一步，只是网络最后是实现分类效果）。然后再根据前面得到的一序列的动作即多个连续时间窗内的源数据利用2层LSTM网络发现在长时间内人们活动的模式，进而做出有效的预测。



****

###程序执行步骤：

#### 源数据为OPPORTUNITY数据集
前期处理借鉴[DeepConvLSTM模型中的处理](https://github.com/sussexwearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb)
针对OPPORTUNITY数据集

```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip
```

```
python preprocess_data.py -h

```

#### 针对torch构造合适数据。

```
python produce.py
```

train/val/test set中的数据已经处理成为```inputs (None, 30,1, 24, 113), targets (None,)```格式

#### 训练

```
th train.lua
```
训练后会将模型保存文件

#### 测试

```
th test.lua
```


