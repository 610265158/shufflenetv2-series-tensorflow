import numbers
import os
import warnings

import numpy as np
import cv2
import random
import math
from train_config import config as cfg

######May wrong, when use it check it
def Rotate_aug(src,angle,label=None,center=None,scale=1.0):
    '''
    :param src: src image
    :param label: label should be numpy array with [[x1,y1],
                                                    [x2,y2],
                                                    [x3,y3]...]
    :param angle:
    :param center:
    :param scale:
    :return: the rotated image and the points
    '''
    image=src
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if label is None:
        for i in range(image.shape[2]):
            image[:,:,i] = cv2.warpAffine(image[:,:,i], M, (w, h),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=0.)
        return image,None
    else:
        label=label.T
        ####make it as a 3x3 RT matrix
        full_M=np.row_stack((M,np.asarray([0,0,1])))
        img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=cfg.DATA.PIXEL_MEAN)
        ###make the label as 3xN matrix
        full_label = np.row_stack((label, np.ones(shape=(1,label.shape[1]))))
        label_rotated=np.dot(full_M,full_label)
        label_rotated=label_rotated[0:2,:]
        #label_rotated = label_rotated.astype(np.int32)
        label_rotated=label_rotated.T
        return img_rotated,label_rotated
def Rotate_coordinate(label,rt_matrix):
    if rt_matrix.shape[0]==2:
        rt_matrix=np.row_stack((rt_matrix, np.asarray([0, 0, 1])))
    full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
    label_rotated = np.dot(rt_matrix, full_label)
    label_rotated = label_rotated[0:2, :]
    return label_rotated


def box_to_point(boxes):
    '''

    :param boxes: [n,x,y,x,y]
    :return: [4n,x,y]
    '''
    ##caution the boxes are ymin xmin ymax xmax
    points_set=np.zeros(shape=[4*boxes.shape[0],2])

    for i in range(boxes.shape[0]):
        points_set[4 * i]=np.array([boxes[i][0],boxes[i][1]])
        points_set[4 * i+1] =np.array([boxes[i][0],boxes[i][3]])
        points_set[4 * i+2] =np.array([boxes[i][2],boxes[i][3]])
        points_set[4 * i+3] =np.array([boxes[i][2],boxes[i][1]])


    return points_set


def point_to_box(points):
    boxes=[]
    points=points.reshape([-1,4,2])

    for i in range(points.shape[0]):
        box=[np.min(points[i][:,0]),np.min(points[i][:,1]),np.max(points[i][:,0]),np.max(points[i][:,1])]

        boxes.append(box)

    return np.array(boxes)


def Rotate_with_box(src,angle,boxes=None,center=None,scale=1.0):
    '''
    :param src: src image
    :param label: label should be numpy array with [[x1,y1],
                                                    [x2,y2],
                                                    [x3,y3]...]
    :param angle:angel
    :param center:
    :param scale:
    :return: the rotated image and the points
    '''

    label=box_to_point(boxes)
    image=src
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心


    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)

    new_size=Rotate_coordinate(np.array([[0,w,w,0],
                                         [0,0,h,h]]), M)

    new_h,new_w=np.max(new_size[1])-np.min(new_size[1]),np.max(new_size[0])-np.min(new_size[0])

    scale=min(h/new_h,w/new_w)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    if boxes is None:
        for i in range(image.shape[2]):
            image[:,:,i] = cv2.warpAffine(image[:,:,i], M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        return image,None
    else:
        label=label.T
        ####make it as a 3x3 RT matrix
        full_M=np.row_stack((M,np.asarray([0,0,1])))
        img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        ###make the label as 3xN matrix
        full_label = np.row_stack((label, np.ones(shape=(1,label.shape[1]))))
        label_rotated=np.dot(full_M,full_label)
        label_rotated=label_rotated[0:2,:]
        #label_rotated = label_rotated.astype(np.int32)
        label_rotated=label_rotated.T

        boxes_rotated = point_to_box(label_rotated)
        return img_rotated,boxes_rotated

###CAUTION:its not ok for transform with label for perspective _aug
def Perspective_aug(src,strength,label=None):
    image = src
    pts_base = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts1=np.random.rand(4, 2)*random.uniform(-strength,strength)+pts_base
    pts1=pts1.astype(np.float32)
    #pts1 =np.float32([[56, 65], [368, 52], [28, 387], [389, 398]])
    M = cv2.getPerspectiveTransform(pts1, pts_base)
    trans_img = cv2.warpPerspective(image, M, (src.shape[1], src.shape[0]))

    label_rotated=None
    if label is not  None:
        label=label.T
        full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
        label_rotated = np.dot(M, full_label)
        label_rotated=label_rotated.astype(np.int32)
        label_rotated=label_rotated.T
    return trans_img,label_rotated

def Affine_aug(src,strength,label=None):
    image = src
    pts_base = np.float32([[10,100],[200,50],[100,250]])
    pts1 = np.random.rand(3, 2) * random.uniform(-strength, strength) + pts_base
    pts1 = pts1.astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts_base)
    trans_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]) ,
                                            borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=cfg.DATA.PIXEL_MEAN)
    label_rotated=None
    if label is not None:
        label=label.T
        full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
        label_rotated = np.dot(M, full_label)
        #label_rotated = label_rotated.astype(np.int32)
        label_rotated=label_rotated.T
    return trans_img,label_rotated
def Padding_aug(src,max_pattern_ratio=0.05):
    src=src.astype(np.float32)
    pattern=np.ones_like(src)
    ratio = random.uniform(0, max_pattern_ratio)

    height,width,_=src.shape

    if random.uniform(0,1)>0.5:
        if random.uniform(0, 1) > 0.5:
            pattern[0:int(ratio*height),:,:]=0
        else:
            pattern[height-int(ratio * height):, :, :] = 0
    else:
        if random.uniform(0, 1) > 0.5:
            pattern[:,0:int(ratio * width), :] = 0
        else:
            pattern[:,width-int(ratio * width):,  :] = 0


    bias_pattern=(1-pattern)*cfg.DATA.PIXEL_MEAN


    img=src*pattern+bias_pattern

    img=img.astype(np.uint8)
    return img

def Blur_heatmaps(src, ksize=(3, 3)):
    for i in range(src.shape[2]):
        src[:, :, i] = cv2.GaussianBlur(src[:, :, i], ksize, 0)
        amin, amax = src[:, :, i].min(), src[:, :, i].max()  # 求最大最小值
        if amax>0:
            src[:, :, i] = (src[:, :, i] - amin) / (amax - amin)  # (矩阵元素-最小值)/(最大值-最小值)
    return src
def Blur_aug(src,ksize=(3,3)):
    for i in range(src.shape[2]):
        src[:, :, i]=cv2.GaussianBlur(src[:, :, i],ksize,1.5)
    return src

def Img_dropout(src,max_pattern_ratio=0.05):
    width_ratio = random.uniform(0, max_pattern_ratio)
    height_ratio = random.uniform(0, max_pattern_ratio)
    width=src.shape[1]
    height=src.shape[0]
    block_width=width*width_ratio
    block_height=height*height_ratio
    width_start=int(random.uniform(0,width-block_width))
    width_end=int(width_start+block_width)
    height_start=int(random.uniform(0,height-block_height))
    height_end=int(height_start+block_height)
    src[height_start:height_end,width_start:width_end,:]=0.

    return src

def Fill_img(img_raw,target_height,target_width,label=None):

    ###sometimes use in objs detects
    channel=img_raw.shape[2]
    raw_height = img_raw.shape[0]
    raw_width = img_raw.shape[1]
    if raw_width / raw_height >= target_width / target_height:
        shape_need = [int(target_height / target_width * raw_width), raw_width, channel]
        img_fill = np.zeros(shape_need, dtype=img_raw.dtype)+np.array(cfg.DATA.PIXEL_MEAN ,dtype=img_raw.dtype)
        shift_x=(img_fill.shape[1]-raw_width)//2
        shift_y=(img_fill.shape[0]-raw_height)//2
        for i in range(channel):
            img_fill[shift_y:raw_height+shift_y, shift_x:raw_width+shift_x, i] = img_raw[:,:,i]
    else:
        shape_need = [raw_height, int(target_width / target_height * raw_height), channel]
        img_fill = np.zeros(shape_need, dtype=img_raw.dtype)+np.array(cfg.DATA.PIXEL_MEAN ,dtype=img_raw.dtype)
        shift_x = (img_fill.shape[1] - raw_width) // 2
        shift_y = (img_fill.shape[0] - raw_height) // 2
        for i in range(channel):
            img_fill[shift_y:raw_height + shift_y, shift_x:raw_width + shift_x, i] = img_raw[:, :, i]
    if label is None:
        return img_fill,shift_x,shift_y
    else:
        label[:,0]+=shift_x
        label[:, 1]+=shift_y
        return img_fill,label




class RandomResizedCrop(object):


    ### torch_convert codes

    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")


        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.shape[1],img.shape[0]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        target_img = img[i:i + h, j:j + w, :]

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
                          cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)

        target_img = cv2.resize(target_img, (self.size[1], self.size[0]), interpolation=interp_method)
        return target_img
class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, target_size,resize_size=256):
        if isinstance(target_size, numbers.Number):
            self.size = (int(target_size), int(target_size))
        else:
            self.size = target_size


        self.resizer=OpencvResize(resize_size)
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        img=self.resizer(img)
        image_width, image_height = img.shape[1],img.shape[0]
        crop_height, crop_width = self.size[0],self.size[1]
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))

        center_croped_img=img[crop_top:crop_top+crop_height,crop_left:crop_left+crop_width,:]

        return center_croped_img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):

        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        img = np.ascontiguousarray(img)

        return img

def box_in_img(img,boxes,min_overlap=0.5):

    raw_bboxes = np.array(boxes)

    face_area=(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])

    h,w,_=img.shape
    boxes[:, 0][boxes[:, 0] <=0] =0
    boxes[:, 0][boxes[:, 0] >=w] = w
    boxes[:, 2][boxes[:, 2] <= 0] = 0
    boxes[:, 2][boxes[:, 2] >= w] = w

    boxes[:, 1][boxes[:, 1] <= 0] = 0
    boxes[:, 1][boxes[:, 1] >= h] = h

    boxes[:, 3][boxes[:, 3] <= 0] = 0
    boxes[:, 3][boxes[:, 3] >= h] = h

    boxes_in = []
    for i in range(boxes.shape[0]):
        box=boxes[i]
        if ((box[3]-box[1])*(box[2]-box[0]))/face_area[i]>min_overlap :
            boxes_in.append(boxes[i])

    boxes_in = np.array(boxes_in)
    return boxes_in

def Random_scale_withbbox(image,bboxes,target_shape,jitter=0.5):

    ###the boxes is in ymin,xmin,ymax,xmax mode
    hi, wi, _ = image.shape

    while 1:
        if len(bboxes)==0:
            print('errrrrrr')
        bboxes_=np.array(bboxes)
        crop_h = int(hi * random.uniform(0.2, 1))
        crop_w = int(wi * random.uniform(0.2, 1))

        start_h = random.randint(0, hi - crop_h)
        start_w = random.randint(0, wi - crop_w)

        croped = image[start_h:start_h + crop_h, start_w:start_w + crop_w, :]

        bboxes_[:, 0] = bboxes_[:, 0] - start_w
        bboxes_[:, 1] = bboxes_[:, 1] - start_h
        bboxes_[:, 2] = bboxes_[:, 2] - start_w
        bboxes_[:, 3] = bboxes_[:, 3] - start_h


        bboxes_fix=box_in_img(croped,bboxes_)
        if len(bboxes_fix)>0:
            break


    ###use box
    h,w=target_shape
    croped_h,croped_w,_=croped.shape

    croped_h_w_ratio=croped_h/croped_w

    rescale_h=int(h * random.uniform(0.5, 1))

    rescale_w = int(rescale_h/(random.uniform(0.7, 1.3)*croped_h_w_ratio))
    rescale_w=np.clip(rescale_w,0,w)

    image=cv2.resize(croped,(rescale_w,rescale_h))

    new_image=np.zeros(shape=[h,w,3],dtype=np.uint8)

    dx = int(random.randint(0, w - rescale_w))
    dy = int(random.randint(0, h - rescale_h))

    new_image[dy:dy+rescale_h,dx:dx+rescale_w,:]=image

    bboxes_fix[:, 0] = bboxes_fix[:, 0] * rescale_w/ croped_w+dx
    bboxes_fix[:, 1] = bboxes_fix[:, 1] * rescale_h / croped_h+dy
    bboxes_fix[:, 2] = bboxes_fix[:, 2] * rescale_w / croped_w+dx
    bboxes_fix[:, 3] = bboxes_fix[:, 3] * rescale_h / croped_h+dy



    return new_image,bboxes_fix


def Random_flip(im, boxes):

    im_lr = np.fliplr(im).copy()
    h,w,_ = im.shape
    xmin = w - boxes[:,2]
    xmax = w - boxes[:,0]
    boxes[:,0] = xmin
    boxes[:,2] = xmax
    return im_lr, boxes


def Mirror(src,label=None,symmetry=None):

    img = cv2.flip(src, 1)
    if label is None:
        return img,None

    width=img.shape[1]
    cod = []
    allc = []
    for i in range(label.shape[0]):
        x, y = label[i][0], label[i][1]
        if x >= 0:
            x = width - 1 - x
        cod.append((x, y))
    # **** the joint index depends on the dataset ****
    for (q, w) in symmetry:
        cod[q], cod[w] = cod[w], cod[q]
    for i in range(label.shape[0]):
        allc.append(cod[i][0])
        allc.append(cod[i][1])
    label = np.array(allc).reshape(label.shape[0], 2)
    return img,label

def produce_heat_maps(label,map_size,stride,sigma):
    def produce_heat_map(center,map_size,stride,sigma):
        grid_y = map_size[0] // stride
        grid_x = map_size[1] // stride
        start = stride / 2.0 - 0.5
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)

        am = np.amax(heatmap)
        if am > 0:
            heatmap /= am / 255.

        return heatmap
    all_keypoints = label
    point_num = all_keypoints.shape[0]
    heatmaps_this_img=np.zeros([map_size[0]//stride,map_size[1]//stride,point_num])
    for k in range(point_num):
        heatmap = produce_heat_map([all_keypoints[k][0],all_keypoints[k][1]], map_size, stride, sigma)
        heatmaps_this_img[:,:,k]=heatmap
    return heatmaps_this_img

def visualize_heatmap_target(heatmap):
    map_size=heatmap.shape[0:2]
    frame_num = heatmap.shape[2]
    heat_ = np.zeros([map_size[0], map_size[1]])
    for i in range(frame_num):
        heat_ = heat_ + heatmap[:, :, i]
    cv2.namedWindow('heat_map', 0)
    cv2.imshow('heat_map', heat_)
    cv2.waitKey(0)


def produce_heatmaps_with_bbox(image,label,h_out,w_out,num_klass,ksize=9,sigma=0):
    heatmap=np.zeros(shape=[h_out,w_out,num_klass])

    h,w,_=image.shape

    for single_box in label:
        if single_box[4]>=0:
            ####box center (x,y)
            center=[(single_box[0]+single_box[2])/2/w,(single_box[1]+single_box[3])/2/h]   ###0-1

            heatmap[round(center[1]*h_out),round(center[0]*w_out),int(single_box[4]) ]=1.

    heatmap = cv2.GaussianBlur(heatmap, (ksize,ksize), sigma)
    am = np.amax(heatmap)
    if am>0:
        heatmap /= am / 255.
    heatmap=np.expand_dims(heatmap,-1)
    return heatmap


def produce_heatmaps_with_keypoint(image,label,h_out,w_out,num_klass,ksize=7,sigma=0):
    heatmap=np.zeros(shape=[h_out,w_out,num_klass])

    h,w,_=image.shape

    for i in range(label.shape[0]):
        single_point=label[i]

        if single_point[0]>0 and single_point[1]>0:

            heatmap[int(single_point[1]*(h_out-1)),int(single_point[0]*(w_out-1)),i ]=1.

    heatmap = cv2.GaussianBlur(heatmap, (ksize,ksize), sigma)
    am = np.amax(heatmap)
    if am>0:
        heatmap /= am / 255.
    return heatmap





if __name__=='__main__':
    pass
