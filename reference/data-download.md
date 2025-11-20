1.	下载
-	Linux
https://support.huaweicloud.com/utiltg-obs/obs_11_0003.html

-	Windows
https://support.huaweicloud.com/browsertg-obs/obs_03_1003.html

2.	配置（可能每次登陆服务器都要重新配置一次，你们可以试试）
-	Linux
用以下命令：
cd /userhome/lingtz/obsutil_linux_amd64_5.3.4/ (修改成解压后的路径)
chmod 755 obsutil
./obsutil config -i=J7EJPN0J3QYK4YVCJWU2 -k=8n4Bw8wk5mMXsRy7XKiM1Hqwm1RYcQPmrJswU3dR -e=obs.cn-south-222.ai.pcl.cn

-	Windows
Windows这个主要是方便你们下载数据到本地，以及可视化文件夹里的文件内容
 
3.	数据上传/下载
-	上传整个文件夹到服务器
./obsutil cp /code/log.log obs://hefuchu-data/westlake-phoenix/ -r -f 

-	上传单个文件到服务器
./obsutil cp /code/casanovo_check obs://hefuchu-data/westlake-phoenix/

-	下载整个文件夹到服务器
./obsutil cp obs://hefuchu-data/westlake-phoenix/1_contrastive/downstream_denovo/casanovo/ /userhome/lingtz/test0911/ -r -f

-	下载单个文件夹到服务器
./obsutil cp obs://hefuchu-data/westlake-phoenix/1_contrastive/downstream_denovo/casanovo/casanovo.py /userhome/lingtz/test0911/

4.	数据位置
-	所有数据都在obs://hefuchu-data/westlake-phoenix/下
-	子文件夹1_contrastive下是和topic1-对比学习预训练模型的下游任务相关的内容
 
-	子文件夹2_seq2seq 下是和topic2的seq-seq预训练模型相关的内容

Note: 因为这个服务器还存着很多其他人的数据，所以注意别误删或告诉其他无关人员。咱们只在obs://hefuchu-data/westlake-phoenix/下操作就好。
多人同时使用可能出现问题，最好安排1-2位同学负责下载就行。



