# Resoure_variation

该项目主要为测量分布式训练任务对边缘设备资源使用情况。

项目需要提前如下依赖：
flower
pytorch
jetson-stats

项目划分为client和server两个部分，server部分为分布式训练中参数服务器的代码，其中参数服务器的启动为
python server.py

