# 3d_detect
=========================================================================================================================
#### 3D检测的思路来自于论文
[3D Bounding Box Estimation Using Deep Learning and Geometry](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mousavian_3D_Bounding_Box_CVPR_2017_paper.pdf)
#### 3D检测的代码是参考了https://github.com/fuenwang/3D-BoundingBox 
#### 2D框是通过[pytorch版本的detectron](https://github.com/roytseng-tw/Detectron.pytorch)得到
思路是把2D检测框输入到3D检测模型进行3D检测，目前能输出目标的长宽高

-------------------------------------------------------------------------------------------------------------------------
### 代码使用方法
#### 首先需要下载权重 
* detectron预训练模型权重 链接: https://pan.baidu.com/s/16XQgYGX-ozUIyFiY7hM9zA 提取码: v2sc 下载完成后放入./data/pretrained_model/下
* detectron我训练好的KITTI权重 链接: https://pan.baidu.com/s/1OWm5qkiCQJ7DbqOLI_AoKA 提取码: qnx4 下载完成后放入./tools/Outputs/e2e_mask_rcnn_R-101-FPN_1x/Dec10-18-16-25_214-2_step/ckpt/ 下
* 3d检测模型权重 链接: https://pan.baidu.com/s/1u-rBhtF4Ngi6MTFFTLzLOQ 提取码: 1fq5 下载完成后放入 ./tools/models/ 下

#### 运行infer_simple _2d-3d.py 参考参数输入 python tools/infer_simple.py --dataset coco --cfg cfgs/baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml --load_ckpt {path/to/your/checkpoint} --image_dir {dir/of/input/images}  --output_dir {dir/to/save/visualizations}

---------------------------------------------------------------------------------------------------------------------------
### 效果图
![](https://github.com/yuyijie1995/3d_detect/blob/master/output1/20180726T144635T6779.png)
三个输出分别是高宽长

--------------------------------------------------------------------------------------------------------------------------
### 训练方法详见 https://github.com/fuenwang/3D-BoundingBox

TODOList：
* 将3D检测框画到2D图上
