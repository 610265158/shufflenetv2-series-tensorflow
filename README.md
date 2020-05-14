# shufflenetv2-series-less

## introduction
A shufflenetv2 and shufflenetv2+ implementations based on tensorflow. 

Shufflenet series is a set of brilliant light DNN models that designed for mobile device mainly.

It keeps slightly better accuracy and higher speed than mobilnet series, 
however there exits no good pretrained model for tensorflow developers.

So the model were converted from the [official pytorch repo](https://github.com/megvii-model/ShuffleNet-Series), with barely no precision loss.

Hope the codes can help you.


## requirment

+ tensorflow1.14    (tensorflow 1.14 at least if mix_precision turns on)

+ pytorch 

+ tensorpack (for data provider)

+ opencv

+ python 3.6

## pretrained model:

+ [baidu disk](https://pan.baidu.com/s/1KwoglosUj_f1NqihlCOAjw)  (code fbur)
+ [google drive](https://drive.google.com/open?id=1yoS5P3cHhD3lO115axoG0aThPeztGvhF)


### performance

* ShuffleNetV2+

| model                  |top1 err       |top5 err  |
| :------:               |:------:       |:------:  |
|  ShuffleNetV2+ Small   | 25.9          |   8.3    |
|  ShuffleNetV2+ Medium  | 0.000x        |0.000x    |  
|  ShuffleNetV2+ Large   | 23.0          |   6.6    |


*  ShuffleNetV2

| model                    |top1 err       |top5 err   |
| :------:                 |:------:       |:------:   |
|  ShuffleNetV2 0.5x	   | 38.9          |17.4       |
|  ShuffleNetV2 1.0x	   | 30.7          |11.2       |
|  ShuffleNetV2 1.5x	   | 27.5          |9.4        |
|  ShuffleNetV2 2.0x	   | 24.9          |7.5        |

Ops, somthing exciting happend when i convert the medium model, 
top1 fucked up, but top5 is fine, 
i thought there is some 
numerical problems .


**Ps, I though that  the other structure listed in the official pytorch repo are not that important for now, 
so i did not do that work. But i will do it when i got time.**

## useage

### train  

Actully you don't need to train them. It has the same params with the official one.

1. download imagenet

2. use this scripts to prepare val set  [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh), 
then run ` python prepare_imagenet.py` produce train.txt and val.txt
(if u like train u own data, u should prepare the data like this:
`path.jpg|label` 

3. download pretrained model from official repo

4. convert pytorch model and modify config

    ```
    example  ShuffleNetV2+.Small
    
    4.1 run`python convert.py --input ShuffleNetV2+.Small.pth.tar 
                              --output ShuffleNetV2+.Small
                              --net_structure ShuffleNetV2_Plus
                              
    4.2 modify config as:
        config.MODEL.net_structure='ShuffleNetV2_Plus'
        config.MODEL.pretrained_model='ShuffleNetV2+.Small.npy'                    ##according to your model,
        config.MODEL.size='Small'     ##Small Medium Large   for v2+
                                      ##0.5x, 1.0x 1.5x 2.0x   for v2
                                      
       
    
    
    example  ShuffleNetV2.0.5x
    
    4.1 run python convert.py --input ShuffleNetV2.0.5x.pth.tar 
                              --output ShuffleNetV2.0.5x
                              --net_structure ShuffleNetV2
                              
    4.2 modify config as:
        config.MODEL.net_structure='ShuffleNetV2'
        config.MODEL.pretrained_model='ShuffleNetV2.0.5x.npy'                    ##according to your model,
        config.MODEL.size='0.5x'     ##Small Medium Large   for v2+
                                      ##0.5x, 1.0x 1.5x 2.0x   for v2
    ```
                              

5. then, run:  `python train.py`， it will save a ckpt model first.

6. ` python tools/auto_freeze.py` convert to pb


### evaluation

1. download the pretrained model from the link before
 
2. `python eval.py --model shufflenet.pb`


### plain use to do classification

`python vis.py --input yourimage.jpg --model yourmodel.pb`



### use as backbone 

```
    features=ShufflenetV2Plus(inputs,training_flag,model_size='Small',include_head=False):
    
    by defaut features is a list contains 4 level features, stride as 4,8,16,32
    
```