i=10000
while(($i<=25000))
do
    python osmn_train_eval.py --data_path /home/jiangsihao/video_seg-master/DAVIS/ --whole_model_path /home/jiangsihao/video_seg-master/models/parent/osmn.ckpt-$i --result_path /home/jiangsihao/video_seg-master/Resultsm17/ --only_testing --data_version 2017 --base_model lite --save_score
    python davis_eval.py /home/jiangsihao/video_seg-master/DAVIS/ /home/jiangsihao/video_seg-master/Resultsm17/ 2017 val 
    let i=i+500
done
i=30000
while(($i<=50000))
do
    python osmn_train_eval.py --data_path /home/jiangsihao/video_seg-master/DAVIS/ --whole_model_path /home/jiangsihao/video_seg-master/models/parent/osmn.ckpt-$i --result_path /home/jiangsihao/video_seg-master/Resultsm17/ --only_testing --data_version 2017 --base_model lite --save_score
    python davis_eval.py /home/jiangsihao/video_seg-master/DAVIS/ /home/jiangsihao/video_seg-master/Resultsm17/ 2017 val 
    let i=i+500
done