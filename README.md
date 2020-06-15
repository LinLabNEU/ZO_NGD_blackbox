# ZO_NGD_blackbox

required packages:

Tensorflow, numpy, PIL



First train models on MNIST and CIFAR:
```
python train_models.py
```

need to download the ImageNet test dataset. 

To obtain ImageNet model:
```
python setup_inception.py
```


Test the performance:
```
python test_attack_mnist.py 
```
or 
```
test_attack_cifar.py 
```
or 
```
test_attack_imagenet.py
```
