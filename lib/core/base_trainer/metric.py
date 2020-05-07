import numpy as np



class Metric():
    def __init__(self,batch_size):

        self.batch_size=batch_size
        self.top1_correct=0
        self.top5_correct = 0
        self.total=0

    def update(self,top1_acc,top5_acc):


        self.top1_correct+=round(top1_acc*self.batch_size)
        self.top5_correct += round(top5_acc * self.batch_size)
        self.total+=self.batch_size


    def report(self):


        ## report
        print('top1 acc:%.6f\n'%(self.top1_correct/self.total))
        print('top5 acc:%.6f\n' % (self.top5_correct / self.total))
        print('%d samples\n'%self.total)
        self.top1_correct = 0
        self.top5_correct = 0
        self.total = 0