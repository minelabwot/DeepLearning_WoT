# DeepLearning_WoT


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
python produce_data.py
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


