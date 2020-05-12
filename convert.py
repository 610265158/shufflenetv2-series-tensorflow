import torch
import numpy as np

params=torch.load('ShuffleNetV2+.Small.pth.tar', map_location=torch.device('cpu'))


params_dict=params['state_dict']



length=len(params_dict)

dict={}



def repalce(word,x,y):

    word=word.replace(x,y)

    return word

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


    if 'first' in k:
        print(params_dict[k].shape)
        print(params_dict[k])


    ### process conv
    if 'weight' in new_k and \
            'fc' not in new_k and \
            'classifier' not in new_k and \
            len(list(params_dict[k].shape))>1:

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

        weights_name_tf=rename(weights_name_tf,'.','/')
        gamma_name_tf = rename(gamma_name_tf, '.', '/')
        beta_name_tf = rename(beta_name_tf, '.', '/')
        moving_mean_tf = rename(moving_mean_tf, '.', '/')
        moving_var_name_tf = rename(moving_var_name_tf, '.', '/')

        weights_name_tf = repalce(weights_name_tf, 'module', 'ShuffleNetV2_Plus')
        gamma_name_tf = repalce(gamma_name_tf, 'module', 'ShuffleNetV2_Plus')
        beta_name_tf = repalce(beta_name_tf, 'module', 'ShuffleNetV2_Plus')
        moving_mean_tf = repalce(moving_mean_tf, 'module', 'ShuffleNetV2_Plus')
        moving_var_name_tf = repalce(moving_var_name_tf, 'module', 'ShuffleNetV2_Plus')
        try:
            if 'depthwise_weights' in weights_name_tf:
                params_dict_tf[weights_name_tf] = np.array(params_dict[weights_name_torch]).transpose(2, 3, 0, 1)
            else:
                params_dict_tf[weights_name_tf]=np.array(params_dict[weights_name_torch]).transpose(2,3,1,0)
        except:
            print(k)
            print(params_dict[weights_name_torch].shape)
        if gamma_name_torch in params_dict:
            params_dict_tf[gamma_name_tf] = np.array(params_dict[gamma_name_torch])
            params_dict_tf[beta_name_tf] = np.array(params_dict[beta_name_torch])
            params_dict_tf[moving_mean_tf] = np.array(params_dict[moving_mean_name_torch])
            params_dict_tf[moving_var_name_tf] = np.array(params_dict[moving_var_torch])





    if 'fc' in new_k or 'classifier' in new_k:
        pre_str, cur_cnt = new_k.rsplit('.', 1)[0].rsplit('.', 1)

        cur_cnt = int(cur_cnt)
        if params_dict[k].shape[1]==1:
            weights_name_torch = pre_str + '.' + str(cur_cnt) + '.' + 'weight'
            weights_name_tf = pre_str + '.' + str(cur_cnt) + '.' + 'depthwise_weights:0'
        else:
            weights_name_torch=pre_str+'.'+str(cur_cnt)+'.'+'weight'
            weights_name_tf=pre_str+'.'+str(cur_cnt)+'.'+'weights:0'






        weights_name_tf=rename(weights_name_tf,'.','/')

        weights_name_tf = repalce(weights_name_tf, 'module', 'ShuffleNetV2_Plus')



        params_dict[weights_name_torch]=np.array(params_dict[weights_name_torch])
        params_dict[weights_name_torch]=np.expand_dims(params_dict[weights_name_torch],-1)
        params_dict[weights_name_torch] = np.expand_dims(params_dict[weights_name_torch], -1)

        try:
            if 'depthwise_weights' in weights_name_tf:
                params_dict_tf[weights_name_tf] = np.array(params_dict[weights_name_torch]).transpose(2, 3, 0, 1)
            else:
                params_dict_tf[weights_name_tf]=np.array(params_dict[weights_name_torch]).transpose(2,3,1,0)
        except:
            print(k)
            print(params_dict[weights_name_torch].shape)



for k,v in params_dict_tf.items():
    if 'first' in k:

        print(v)
        print(k,v.shape)


np.save('shufflenetv2plus.npy',params_dict_tf)



cheker=np.load('shufflenetv2plus.npy',allow_pickle=True).item()

# for k,v in cheker.items():
#     print(k,v.shape)
