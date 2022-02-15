# Resoure_variation

该项目主要为测量分布式训练任务对边缘设备GPU资源利用使用情况。

项目需要提前如下依赖：
flower
pytorch
jetson-stats

项目划分为client和server两个部分，server部分为分布式训练中参数服务器的代码，其中参数服务器的启动为

python3 server.py

client部分为分布式训练中worker的代码，其中worker启动和测量worker边缘设备的GPU资源利用情况的脚本为

./client_start.sh

当分布式训练任务完成后可以直接终止脚本运行，脚本将输出worker边缘设备的GPU资源利用情况文件 log_training.csv
