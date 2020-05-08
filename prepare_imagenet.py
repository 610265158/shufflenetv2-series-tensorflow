#-*-coding:utf-8-*-

import os

data_set_dir='./IMAGENET'
train_dir=os.path.join(data_set_dir,'train')
val_dir=os.path.join(data_set_dir,'val')

labels=os.listdir(train_dir)



##filter
labels=[x for x in labels if os.path.isdir(os.path.join(train_dir,x))]



labels.sort()


syntext_f=open('label.txt','w')
for label in labels:
    message = label + ' ' +str(labels.index(label)) +'\n'
    syntext_f.write(message)


train_f=open('train.txt','w')
val_f=open('val.txt','w')



for label in labels:
    cur_dir=os.path.join(train_dir,label)

    pic_list=os.listdir(cur_dir)

    for pic in pic_list:
        cur_path=os.path.join(cur_dir,pic)


        message=cur_path+'|'+str(labels.index(label))+'\n'
        train_f.write(message)

for label in labels:
    cur_dir = os.path.join(val_dir, label)

    pic_list = os.listdir(cur_dir)

    for pic in pic_list:
        cur_path = os.path.join(cur_dir, pic)

        message = cur_path + '|' + str(labels.index(label)) + '\n'
        val_f.write(message)



train_f.close()
val_f.close()
syntext_f.close()