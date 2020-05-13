# shufflenetv2

## introduction
A shufflenetv2 implementation based on tensorflow. 

pretrained model:

+ [baidu disk](https://pan.baidu.com/s/1jPW9cq9V9sJDrcrtcqpmLQ)  (code wd5g)
+ [google drive](https://drive.google.com/open?id=1YHtaLkalAqURbkIYYJBLf6HJZzd6vzOG)


## requirment

+ tensorflow1.14    (tensorflow 1.14 at least if mix_precision turns on)

+ pytorch 

+ tensorpack (for data provider)

+ opencv

+ python 3.6


### performance

then trained tensorflow 
ShuffleNetV2+

| model                  |top1 acc      |top5 acc|
| :------:               |:------:      |:------:  |
|  ShuffleNetV2+ Small   | 0.735        |0.912|

ShuffleNetV2

| model                    |top1 acc      |top5 acc|
| :------:                 |:------:      |:------:  |
|  ShuffleNetV2 0.5x	   | 0.601        |0.82|

## useage

### train

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
    
    
**ps, the model precision is not keeped after convert, 
and i thought, it is because the label order and some difference between pytorch and tensorflow,
but it is fine after finetune the model with about 150,000 iters**
                              

5. then, run:  `python train.py`


### evaluation

1. better convert the model to pb first by run `python tools/auto_freeze.py`,
 it reads from the last checkpint
 
2. `python eval.py --model shufflenet.pb`