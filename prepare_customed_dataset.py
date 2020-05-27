#-*-coding:utf-8-*-

import os

import random


ratio=0.8
data_set_dir='./IMAGENET'
def prepare_data():



    labels=os.listdir(data_set_dir)


    ##filter
    labels=[x for x in labels if os.path.isdir(os.path.join(data_set_dir,x))]

    labels.sort()


    syntext_f=open('label.txt','w')
    for label in labels:
        message = label + ' ' +str(labels.index(label)) +'\n'
        syntext_f.write(message)


    train_f=open('train.txt','w')
    val_f=open('val.txt','w')

    for label in labels:
        cur_dir=os.path.join(data_set_dir,label)

        pic_list=os.listdir(cur_dir)


        random.shuffle(pic_list)


        num_data=len(pic_list)

        train_list=pic_list[:int(ratio*num_data)]
        val_list = pic_list[int(ratio*num_data):]
        for pic in train_list:
            cur_path=os.path.join(cur_dir,pic)

            message=cur_path+'|'+str(labels.index(label))+'\n'
            train_f.write(message)

        for pic in val_list:
            cur_path = os.path.join(cur_dir, pic)

            message = cur_path + '|' + str(labels.index(label)) + '\n'
            val_f.write(message)


    train_f.close()
    val_f.close()
    syntext_f.close()


if __name__=='__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", required=False, default=0.8,type=float, help="train val split ratio")
    ap.add_argument("--datadir", required=False, default="IMAGENET", help="train val split ratio")

    args = ap.parse_args()

    ratio = args.ratio
    data_set_dir=args.datadir


    prepare_data()