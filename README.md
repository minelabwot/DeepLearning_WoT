### 基本介绍：

### introduction

Rapid increase in connectivity of physical sensors and Internet of Things (IoT) systems is enabling large-scale collection of time series data, and the data represents the working patterns and internal evolutions of observed objects. Recognizing and forecasting the underlying high-level  states from raw sensory data are useful for daily activity recognition of humans and  predictive maintenance of machines. Deep Learning (DL) methods have been proved efficient in computer vision, natural language processing, and speech recognition, and these model are also applied to time series analysis. 

Many works has been done to improve the performance of the feature extraction from time series data, but they mostly focus on getting a better vector representation of the time series data from a time window. So a hybrid deep architecture named Long-term Recurrent Convolutional LSTM Network (LR-ConvLSTM) is proposed. The model is composed of Convolutional LSTM layers to extract features inside a high-level state, and extra LSTM layers to capture temporal dependencies between high-level states.

***

### architecture

* model：

![](model1.png)

* 模型介绍：

Our model has three major components: the convolutional layers, the inner LSTM layers and the outer LSTM layers, stacked from bottom to top. The CNN and inner LSTMs are combined to capture features from a sliding window, and the performance of this united model has been proved by many works. After that we use the outer LSTMs to capture long temporal dependencies to play a corrective role in the pattern recognition tasks.


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

**Note**此处需要针对不同的输入，来做相应的调整：比如做针对```sliding_window_length```不同的对比试验时，就要改变文件中的该参数，生成对应的数据集以供训练和测试。

模型输入格式为```(activity_length,sliding_window_length,1,signal_channels)```

调整好输入后，运行该文件生成数据集。

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

#### 结果与分析

当```sliding_window_length```为24时，基于20个动作序列的历史信息，分别做识别与预测任务(预测此后一步)，结果如下：

|model|classification|prediction|
|---|---|----|
|LR-ConvLSTM|86.64|74.39|
|LRCN|79.13|74.25|
|deepConvLSTM|81.45|null|

有待更新。。。


