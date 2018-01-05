
Here is a simple process for using the depth model in imbalanced data.

As we all know, the generalization of model is an important measure. So we also evaluated the LR-convLSTM model in an industrial dataset that records the true operating status of the devices which provided by PHM 2015 challenge. At the same time, This dataset is quite imbalanced as the data in fault state is very rare compared to the data in normal mode. So recognizing the operating status of the devices and identifing what kind of fault the data characterize is difficult. 

an overview of the proportion of different state types in the data set:

|Fault1|Fault2|Fault3|Fault4|Fault5|Fault6|Normal|
|---|---|---|---|---|---|---|
|4.83%|3.75%|3.39%|0.06%|0.65%|19.26%|68.06%|

In this extremely balanced data set, any model to complete the fault classification task is a huge challenge. 

#### The problem of machine learning methods in imbalanced data

The goal of the optimization of machine learning methods is to minimize the total loss for training dataset. But in extremely imbalanced dataset( such as the data of fault state is only 1%), even if all the data is identified as normal state, the lassification rate is still up to 99%. This is not the result we want. The correctly recognition of the fault state is more valuable to us in such condition.

More detailed, each sample is considered equally important in the machine learning methods by default which leads to the result that the depth model is biased in favor of the majority class.

#### the idea for improvement -- adjust the weight of each class in the loss function

So one idea for improvement is to make the loss of each category the same importance. That is multiplying a large weight for the minority class of data, majority data multiplied by a small weight. Such parameters are provided in many deep learning frameworks. In Torch, The negative log likelihood (NLL) criterion `criterion = nn.ClassNLLCriterion([weights, sizeAverage, ignoreIndex])`,[If provided, the optional argument weights should be a 1D Tensor assigning weight to each of the classes. This is particularly useful when you have an unbalanced training set.](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion)

> The loss can be described as:

> 	loss(x, class) = -x[class]

> or in the case of the weights argument, it is specified as follows:

> 	loss(x, class) = -weights[class] * x[class]

here, the `weights[class]=1/NUMBER[class]`

#### intrgrated with manual data preprocess method

However, we found that the practice is not always according to it. Only adding a weight to criterion is not enough to make the training result stable. So we need to use the traditional data preprocessing methods to reduce the imbalance between data classes lightly. 

*data preprocessing methods for imbalanced data*

* over-sample
	SMOTE method is a typical method used for over sampling data.

* under-sample
	random down sampling is a typical method of under-sample.

Considering that the over-sample method is relatively easier to form overfitting, we use under-sample method in most cases.

[a simple process for using the depth model in imbalanced data](normalDetect.png)

* ❤️ Hopefully, those ideas can help serve as a guide to all of you as you continue along this process. ❤️*
