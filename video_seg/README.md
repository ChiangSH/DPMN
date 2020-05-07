# DPMN:基于OSMN的改进视频目标分割算法

OSMN参考如下引用信息:
```
@article{ Yang2018osmn,
  author = {Linjie Yang and Yanran Wang and Xuehan Xiong and Jianchao Yang and Aggelos K. Katsaggelos},  
title = {Efficient Video Object Segmentation via Network Modulation},
  journal = {CVPR},
  year = {2018}
}
```
#In this work, we propose to use a meta neural network named modulator to manipulate the intermediate layers of the segmentation network given the appearance of the object in the first frame. Our method only takes 140ms/frame for inference on DAVIS dataset.

#<img src='doc/ims/model_structure.png'>

## 安装
1. 下载代码命令如下
   ```Shell
   git clone https://github.com/ChiangSH/DPMN.git
   ```
2. 安装环境如下:
   
   - Python 3.5 
   - Tensorflow r1.0 or higher (`pip install tensorflow-gpu`) along with standard [dependencies](https://www.tensorflow.org/install/install_linux)
   - Densecrf by [Philipp Krähenbühl and Vladlen Koltun](https://github.com/lucasb-eyer/pydensecrf)
   - Other python dependencies: PIL (Pillow version), numpy, scipy
   

## 离线训练

### Stage 1: 在MS-COCO上进行第一步离线训练
1. 下载 MS-COCO 2017 数据集 [here](http://cocodataset.org/#download).
2. 下载 VGG 16 模型 [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).
3. 将 vgg_16.ckpt 文件放入 `models/`.
4. 执行命令 `python osmn_coco_pretrain.py --data_path DATA_PATH --model_save_path MODEL_SAVE_PATH --gpu_id GPU_ID --training_iters 200000` ，默认学习率为 1e-5 ；
   然后执行命令 `python osmn_coco_pretrain.py --data_path DATA_PATH --model_save_path MODEL_SAVE_PATH --gpu_id GPU_ID --training_iters 300000 --learning_rate 1e-6` 来进行更深的训练；
   确保 `MODEL_SAVE_PATH` 的路径一致
   参数解释可执行命令 `python osmn_coco_pretrain.py -h`.

### Stage 2: 在DAVIS上进行微调离线训练
1. 下载DAVIS 2017 数据集 [here](http://davischallenge.org/code.html).
2. 执行数据预处理命令 `python preprocessing/preprocess_davis.py DATA_DIR`.
3. 执行命令 `python osmn_train_eval.py --data_path DATA_PATH --whole_model_path WHOLE_MODEL_PATH --result_path RESULT_PATH --model_save_path MODEL_SAVE_PATH_FT --gpu_id GPU_ID --batch_size 4 --fix_bn --randomize_guide --training_iters 50000 --learning_rate 1e-6`.
 这里的 `WHOLE_MODEL_PATH` 与 Stage 1 一致. 结果保存在 `RESULT_PATH`.

### 在DAVIS上进行在线微调
执行命令
```
python osmn_online_finetune.py --whole_model_path WHOLE_MODEL_PATH --result_path RESULT_PATH --model_save_path MODEL_SAVE_PATH_OL --gpu_id GPU_ID --batch_size 1 --training_iters 100 --data_version [2016/2017]
```

## 评估
1. 执行生成测试结果命令
```
python osmn_train_eval.py --data_path DATA_PATH --whole_model_path WHOLE_MODEL_PATH --result_path RESULT_PATH --only_testing --data_version [2016/2017] --gpu_id GPU_ID [--save_score] [--use_full_res]
```
`--save_score` 仅在 DAVIS 2017测试时使用。`--use_full_res` 表示使用全分辨率的图片来进行测试。
2. 得到mIOU结果：
```
python davis_eval.py DATA_PATH RESULT_PATH DATASET_VERSION DATASET_SPLIT
```

