from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/Library')
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange
import time
# Use a non-interactive backend
from torchvision.models.vgg import *
import matplotlib
matplotlib.use('Agg')
import yaml
from torch.autograd import Variable

import Model
import Dataset
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset',default='COCO')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args

def GetImage(imagepath):

    img = cv2.imread(imagepath, cv2.IMREAD_COLOR).astype(np.float) / 255
    img[:, :, 0] = (img[:, :, 0] - 0.406) / 0.225
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.485) / 0.229
    return img

def threeD_detect(img_path,bboxes,i):
    class_dict={0:'background',1:'Car',2:'Cyclist',3:'Pedestrian'}
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/models'
    if not os.path.isdir(store_path):
        print('No folder named \"models/\"')
        exit()

    model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]

    if len(model_lst) == 0:
        print('No previous model found, please check it')
        exit()
    else:
        print('Find previous model %s' % model_lst[-1])
        vgg = vgg19_bn(pretrained=False)
        model = Model.Model(features=vgg.features, bins=2).cuda()
        params = torch.load(store_path + '/%s' % model_lst[-1])
        model.load_state_dict(params)
        model.eval()

    img=GetImage(img_path)
    all_results=[]
    for bbox in bboxes:
        batch = np.zeros([1, 3, 224, 224], np.float)
        xmin=int(bbox[0])
        ymin=int(bbox[1])
        xmax=int(bbox[2])
        ymax=int(bbox[3])

        crop = img[ymin:ymax + 1, xmin:xmax + 1]
        crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        batch[0, 0, :, :] = crop[:, :, 2]
        batch[0, 1, :, :] = crop[:, :, 1]
        batch[0, 2, :, :] = crop[:, :, 0]


        batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()

        [orient, conf, dim] = model(batch)
        # orient = orient.cpu().data.numpy()[0, :, :]
        # conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]
        dim=dim.tolist()
        result=[xmin,ymin,xmax,ymax,dim,class_dict[i]]
        all_results.append(result)
    return all_results

def main():
    """main function"""

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if args.cuda:
        maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()

    params=list(maskRCNN.parameters())
    k=0
    for i in params:
        l=1
        for j in i.size():
            l*=j
        k=k+l
    print('zonghe:'+str(k))

    if args.image_dir:
        imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
    else:
        imglist = args.images
    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i in xrange(num_images):
        print('img', i)
        im = cv2.imread(imglist[i])
        assert im is not None

        timers = defaultdict(Timer)
        start=time.time()
        cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)
        class_result_boxes=[]
        for index,class_boxes in enumerate(cls_boxes):
            if len(class_boxes) !=0:
                class_boxes=class_boxes.tolist()
                results_oneclass=threeD_detect(imglist[i],class_boxes,index)
                class_result_boxes.append(results_oneclass)
        save_image=im
        color_class={'Car':[0,255,255],'Cyclist':[255,0,0],'Pedestrian':[0,0,255]}
        for result_boxes in class_result_boxes:
            for box in result_boxes:
                cv2.rectangle(save_image,(box[0],box[1]),(box[2],box[3]),color_class[box[-1]],2)
                height=round(box[-2][0],2)
                width=round(box[-2][1],2)
                length=round(box[-2][2],2)
                threeD_info=str(height)+' '+str(width)+' '+str(length)
                cv2.putText(save_image,threeD_info,(box[0],box[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                _,imagename=os.path.split(imglist[i])
                imagename2=imagename.split('.')[0]
                cv2.imwrite('../output1/%s.png'%imagename2,save_image)

        end=time.time()
        print(end-start)
        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )

    if args.merge_pdfs and num_images > 1:
        merge_out_path = '{}/results.pdf'.format(args.output_dir)
        if os.path.exists(merge_out_path):
            os.remove(merge_out_path)
        command = "pdfunite {}/*.pdf {}".format(args.output_dir,
                                                merge_out_path)
        subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
