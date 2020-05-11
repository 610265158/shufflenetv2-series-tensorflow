import torch
import numpy as np

params=torch.load('ShuffleNetV2+.Small.pth.tar',map_location=torch.device('cpu') )


params_dict=params['state_dict']



length=len(params_dict)

dict={}


def rename(word,x,y):



    res=''
    for i,item in enumerate(word):
        if item=='.':
            res+='/'
        else:
            res+=item

    return res


def rename(word,x,y):
    res=''
    for i,item in enumerate(word):
        if item=='.':
            res+='/'
        else:
            res+=item

    return res
params_dict_tf={}


cnt=0
for k,v in params_dict.items():

    new_k=k


    print(k,v.shape)




    ### process conv
    if 'weight' in new_k and 'fc' not in new_k and len(list(params_dict[k].shape))>1:

        ### conv weights

        pre_str,cur_cnt=new_k.rsplit('.',1)[0].rsplit('.',1)

        cur_cnt=int(cur_cnt)

        ### conv weight name

        if params_dict[k].shape[1]==1:
            weights_name_torch = pre_str + '.' + str(cur_cnt) + '.' + 'weight'
            weights_name_tf = pre_str + '.' + str(cur_cnt) + '.' + 'depthwise_weights:0'
        else:
            weights_name_torch=pre_str+'.'+str(cur_cnt)+'.'+'weight'
            weights_name_tf=pre_str+'.'+str(cur_cnt)+'.'+'weights:0'


        ## gamma =
        try:
            gamma_name_torch = pre_str + '.' + str(cur_cnt+1) + '.' + 'weight'
            gamma_name_tf = pre_str + '.' + str(cur_cnt) + '.' + 'BatchNorm/gamma:0'

            ##beta
            beta_name_torch = pre_str + '.' + str(cur_cnt + 1) + '.' + 'bias'
            beta_name_tf = pre_str + '.' + str(cur_cnt) + '.' + 'BatchNorm/beta:0'


            ##movin mean
            moving_mean_name_torch = pre_str + '.' + str(cur_cnt + 1) + '.' + 'running_mean'
            moving_mean_tf = pre_str + '.' + str(cur_cnt) + '.' + 'BatchNorm/moving_mean:0'
            ##movin var
            moving_var_torch = pre_str + '.' + str(cur_cnt + 1) + '.' + 'running_var'
            moving_var_name_tf = pre_str + '.' + str(cur_cnt) + '.' + 'BatchNorm/moving_variance:0'



            params_dict_tf[weights_name_tf]=np.array(params_dict[weights_name_torch]).transpose(2,3,0,1)
            params_dict_tf[gamma_name_tf] = np.array(params_dict[gamma_name_torch])
            params_dict_tf[beta_name_tf] = np.array(params_dict[beta_name_torch])
            params_dict_tf[moving_mean_tf] = np.array(params_dict[moving_mean_name_torch])
            params_dict_tf[moving_var_name_tf] = np.array(params_dict[moving_var_torch])
        except:
            print('this weights has no bn')





for k,v in params_dict_tf.items():
    print(k,v.shape)