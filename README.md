# face_landmark
A simple face aligment method


## introduction
A shufflenetv2 implementation based on  tensorflow. 

pretrained model:

+ [baidu disk](https://pan.baidu.com/s/1jPW9cq9V9sJDrcrtcqpmLQ)  (code wd5g)
+ [google drive](https://drive.google.com/open?id=1YHtaLkalAqURbkIYYJBLf6HJZzd6vzOG)


## requirment

+ tensorflow1.14    (tensorflow 1.14 at least if mix_precision turns on)

+ pytorch 

+ tensorpack (for data provider)

+ opencv

+ python 3.6


## useage

### train

1. download imagenet 

2. run ` python prepare_imagenet.py` produce train.txt and val.txt
(if u like train u own data, u should prepare the data like this:
`path.jpg|label` 
3. download pretrained model

4. then, run:  `python train.py`