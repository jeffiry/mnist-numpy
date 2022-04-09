# mnist-numpy

### Dataset
trainset.npz testset.npz should be in the folder 'datasets'
The dataset is in the https://pan.baidu.com/s/1ARj9JDSNUgUydHt8wBKosA?pwd=cfpk

### Use
python mnist.py
And then input the lr, h, l2parameter. Such as '0.005 400 0.00005'
If the model has exited, the model will be loaded.
If not, a new model will be trained and tested.

In folder 'dnn', there is layers, net, optimizer and loss functions.
In folder 'models', there is some trained models.
