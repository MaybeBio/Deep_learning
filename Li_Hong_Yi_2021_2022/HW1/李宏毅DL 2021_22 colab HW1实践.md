**总体学习规划：2021/2022年的课程内容学完——》2023年的生成式AI也就是LLM随时补充**

也就是按照B站网课视频学完21/22+23年的内容，

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728041305260-96f1ca87-6b35-433d-bbfb-f20aa3e4ce9a.png)

然后追更24年的生成式AI

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728041353096-6937a7bd-002f-4529-9997-765974bc7b6c.png)



从今天起开一个新坑：  
就是李宏毅的深度学习课程，为了参考资料的完整性，主要是参考2021/2022 SPRING时期的网课，

参考B站视频[https://www.bilibili.com/video/BV1Wv411h7kN/?spm_id_from=333.337.search-card.all.click&vd_source=00f11bdb0cf4cafcaf7d8413135e5bb7](https://www.bilibili.com/video/BV1Wv411h7kN/?spm_id_from=333.337.search-card.all.click&vd_source=00f11bdb0cf4cafcaf7d8413135e5bb7)，

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727872072110-98dbd8fe-e649-4dbf-b0e5-a86541fa7415.png)

实际视频中为了补全资料之间的完整性以及整个学习体系的结构性，很多视频不仅仅限制于21以及22年的课件，也补充了很多李宏毅老师之前的课件，主要是2021/22的spring以及2017的fall，可以找到该视频中大部分对应内容的课件：

[https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727872230745-89417c71-fb56-4dee-a8b3-c7167b161298.png)

所以在实际自学过程中，可以完全按照视频的内容顺序进行，然后在台大网站上找到对应课件资料即可（一般补充视频的课件资料大都是几年前的课程）

**总之此处以22年spring的课件大纲为主线，然后作业homework以及原始代码的话都可以在原网页中找到：**

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727872380594-642501e8-a02d-4ab0-b8b5-0a82eebe5969.png)

HW1的代码参考课件官网中的网址为：

[https://colab.research.google.com/drive/1FTcG6CE-HILnvFztEFKdauMlPKfQvm5Z#scrollTo=YdttVRkAfu2t](https://colab.research.google.com/drive/1FTcG6CE-HILnvFztEFKdauMlPKfQvm5Z#scrollTo=YdttVRkAfu2t)

另外注释：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727872767345-cfa6c1dc-a910-43ca-beab-0d4a05006c8f.png)

[https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb](https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)

github中脚本代码是原始参考，官网中给出的脚本代码是HW1的修改基础（也就是蓝图）  
所有的操作都是基于官网给出的代码，但是在修改微调的时候可以参考github中的原始脚本





**HW1的任务：**  
简单来说是个regression回归任务

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727970308792-4c9cc64a-acaf-4854-b04d-db786ac18c2b.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727970321338-3f467e71-0101-4f85-b5ff-da66bb5e47fc.png)

构建模型过程中其实完全不用考虑test测试集，所以只要看train训练集即可；

train训练集中提供了2688名sample（人）x每sample118个feature的数据，

这118个feature包括第1列的ID列，37个state的one-hot编码，5天的阳性率检测相关数据（每天有上面ppt中4+8+3的表型数据feature共15列，然后阳性率的1列，即每天都有16个feature检测，然后每个样本一共检测了5天）

在实际训练过程中，实际上每个样本只获取其前117个feature进行train，即第5天的最后一列feature阳性率作为label不纳入train中，用于label-expected value来计算loss，优化模型架构；

然后train数据中5天数据训练出来的模型，最后用于评估test数据集中第5天的阳性率（test测试集同样只提供了前117维feature，同时注意train和test是完全不一样的数据，ID列具有迷惑性但是不是同一数据，总之就是完全不相关）

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728017662056-f172ace9-22cf-41fd-8618-3f078f1496ef.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728018933496-0b7f1647-4d24-47f5-ba30-f0f57eb09aba.png)



1，首先是对于模型数据的查看：包括训练集以及测试集

训练集：  
实际上经过检查（有数据id缺失）是2699x118，

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727871386704-0f3070d0-5a73-4d07-9f39-b27750620e6e.png)

然后在列上：

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727871409384-482d8be4-5de5-4dc9-8902-b79e6cf10f50.png)

第1列是id，紧接着是所在美国state所在地的one-hot编码，

one-hot之后是其他的一些feature，然后这些feature重复了5轮，实际上就是5天的data

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727871428046-86e09a8b-c6c5-458e-a697-9a22a9dbe8d8.png)

每轮最后一列其实就是label值，也就是感染的阳性率

![](https://prod-files-secure.s3.us-west-2.amazonaws.com/8d8073a7-3313-4a71-8014-f771694a15d4/6cea9fdc-c2d6-4b39-a5f2-9ccc01d13431/image.png)![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727871439242-7c1f6299-5509-40ff-adc8-5cf83524b93a.png)



至于测试集：  
实际上只有1078x117，也就是样本数少了，feature的话同样是给出了这些ID的前4天label值，但是第5天缺省label

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727871456119-bcac0292-8ee9-4734-855e-6905e616ca5c.png)

注意到虽然ID这一列很有迷惑性，但是ID相同的行实际上其他数据完全不一样，也就是说train和test是完全不同的数据，并没有11对应的关系

2，colab上代码实际执行过程中：因为训练的时间比较长，可能在训练过程中会中断，所以参考一些java脚本操作来保证运行过程中模型训练不会中断：  
参考：

[https://blog.csdn.net/Thebest_jack/article/details/124565741](https://blog.csdn.net/Thebest_jack/article/details/124565741)

以及[https://gist.github.com/e96031413/80a633f6a07c150b431639a4e3c606a8](https://gist.github.com/e96031413/80a633f6a07c150b431639a4e3c606a8)

主要是参考前者

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727871489221-68e29616-1052-4d38-95d6-df96cfe814d8.png)

当然实际执行过程中如果神经网络已经训练完成，那么colab连接也会中断（所以情况主要是用于模型训练比较花费时间的过程）

3，如果不改动助教的代码过程，即按照原始的脚本执行过程：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727873185443-3c914833-eff6-4e30-a010-b97b66f3fb1d.png)

如果使用tensorboard查看train以及valid过程中的loss就是：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727873670935-07ec52b2-2649-4afc-9deb-c553b085dce3.png)

train维持在1.8左右，valid维持在2.0左右

最后对test数据集进行预测就是：

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727874812810-2c4deaf2-50e2-4504-8ee5-dc6e6fc4c28d.png)

也就是1078个人的第5天的label

4，所有的ipynb脚本保存到github上：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727876278896-835f703f-4891-40f5-bc08-97cb6b1d68c1.png)

**官网上的脚本是*_original

官网参考过的脚本是*_official

自己注释过的是*_annotated

然后大佬复现学习的脚本就按照*_medium等等级来记录**





每个作业，官方的脚本仅仅作为baseline（实际上就是所有作业代码的base）

2021的官方脚本参考[https://github.com/ga642381/ML2021-Spring](https://github.com/ga642381/ML2021-Spring)

2022的官方脚本参考[https://github.com/virginiakm1988/ML2022-Spring](https://github.com/virginiakm1988/ML2022-Spring)

然后改进/修改的方向，也就是各位开源贡献者大佬的代码，是用于复现以及内化吸收的部分：  
主要参考来源：

（1）[https://github.com/yaoweizhang/LHY2022-SPRING/tree/main](https://github.com/yaoweizhang/LHY2022-SPRING/tree/main)  120stars

主要是其中的各个层级的baseline（都要看看）——>按照修改的顺序进行内化学习

也就是simple——》medium——》strong——》boss baseline的顺序

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727878675061-b4f28887-c60c-42e2-9038-93f9624001d2.png)

（2）[https://github.com/Hoper-J/HUNG-YI_LEE_Machine-Learning_Homework](https://github.com/Hoper-J/HUNG-YI_LEE_Machine-Learning_Homework) 121stars

（3）[https://github.com/wolfparticle/machineLearningDeepLearning](https://github.com/wolfparticle/machineLearningDeepLearning) 530stars

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1727878962491-ed00a601-a60d-487f-8ebc-3b5ca8510c78.png)

但这里主要是2021年的，所以仅仅作为参考，略

（4）[https://github.com/sotaBrewer824/LHY_MLDL/tree/main](https://github.com/sotaBrewer824/LHY_MLDL/tree/main) 15stars

**之后作业代码的训练都以（1），（2），（4）为主要参考，（3）略

以及每一个HW在simple、medium、strong、boss baseline上都要试一试**

5，对model进行的优化

主要参考上面的（1），即[https://github.com/yaoweizhang/LHY2022-SPRING/tree/main/Hw01/answer](https://github.com/yaoweizhang/LHY2022-SPRING/tree/main/Hw01/answer)

**之后的代码优化参考主要都以（1）为主！！！**

然后就是具体的优化目标：各个baseline

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728017572566-ee445d1f-a105-492d-a0d0-a1603b6ebf13.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728017493208-45c6e662-ac72-458c-8452-370afb605616.png)

（1）medium baseline：  
主要就是依据domain knowledge精挑细选一些有用的feature

依据[https://github.com/wolfparticle/machineLearningDeepLearning/blob/main/homework_code/hw1/%E4%BD%9C%E4%B8%9A%E8%AF%B4%E6%98%8E.txt](https://github.com/wolfparticle/machineLearningDeepLearning/blob/main/homework_code/hw1/%E4%BD%9C%E4%B8%9A%E8%AF%B4%E6%98%8E.txt)

即[https://delphi.cmu.edu/covidcast/survey-results/?date=20210221](https://delphi.cmu.edu/covidcast/survey-results/?date=20210221)与[https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/fb-survey.html#mental-health-indicators](https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/fb-survey.html#mental-health-indicators)，

主要就是对feature的一些解读，

此处主要是修改一些feature：  
simple中是使用了全部的feature（包括ID列）

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728018388915-cf2fc173-f766-42cf-a403-22eb4009aea0.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728018434124-d32d16f8-ee82-4700-be62-feff85bc7a00.png)

medium中并没有使用全部的feature列，也就是false循环分支，但是直接使用原本的[0,1,2,3,4]列也未免太过于随意，

实际上就是选了id列+前4个state名字

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728018572667-1970ded9-09cb-4d26-a740-4c0cd1935e57.png)

而medium baseline中改为了

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728018682100-7565b5b5-f45b-4f56-9289-600ce6c2f66b.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728019735871-4b38c078-c8f5-4202-8708-3c45a583ac51.png)

也就是说该medium baseline认为前4天的阳性率是比较重要的feature，对于预测第5天的阳性率有用（当然我们可以修改为其他的feature，比如说第5天的一些表型数据+前4天的阳性率数据等，这个在后面的其他更高级的baseline中也有所修改）

实际操作的时候：

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728020420385-e8059431-8296-44f6-863a-1688d326b0ce.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728020493826-0a17f350-4b4b-415f-9df6-01187d02f468.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728020537919-b12610c2-7962-4c4b-a51f-d8fc7a4fe8f5.png)

**我们知道是select_feat函数的select_all参数要决定使用全部feature还是自己挑出来的feature，

而这个参数可以在config中修改，通过config['select_all']在data加载时传入；也可以不修改config，直接在data加载时在select_feat函数中直接定义False值；

个人建议在config中修改，这样与其他的参数配置统一化**

  
示例中medium并没有在config中修改，而是直接传入false参数

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728020337433-84ff6d62-a3a6-44ca-a61b-60a26111b030.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728020827956-98d2093f-afe5-4b33-8d7f-a83d7959623e.png)



本人同理修改如下：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728020804786-e21cd73d-cf8d-4581-8b07-06b4fdec66a6.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728020815487-237aaf4b-e7c8-40e1-8729-7d2b675233d4.png)

最后训练结果：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728022206611-bff0d1ba-e7db-40b5-9c3b-e306a3961531.png)

可以看得出来train和valid的loss相比较simple都下降了不少

train维持在1.2左右，valid维持在1.3左右

与参考一致：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728022224180-8a533dc6-c49d-4cf3-abd7-6f8ce88af52c.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728023407996-5f81105f-78eb-425e-bac2-089ad41c7736.png)

但是每次都是分开的图，查看实在是太不直观了，里路上train以及valid是可以放在一起比较的，可视化效果也更好，而且实际操作也就是折线图叠加（没什么技术难度）

即参考[https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb](https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)

**todo：

将train、valid的loss整合在同一幅图中比较可视化；

valid中label与expected_value预测值的比较**

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728025598133-07368368-273c-46a2-8297-0da4fafe9f5c.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728025619550-e63eb5ac-5032-4adc-b1f9-2c5ef2430993.png)

（2）strong baseline:  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728017572566-ee445d1f-a105-492d-a0d0-a1603b6ebf13.png?x-oss-process=image%2Fformat%2Cwebp)![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728017493208-45c6e662-ac72-458c-8452-370afb605616.png?x-oss-process=image%2Fformat%2Cwebp)

依据参考主要是在nn网络结构上优化，以及使用除了SGD随机梯度下降之外其他的优化算法；

首先是nn网络结构的修改，将

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728028448832-8de3b8bc-c3c3-46c1-984a-26a7d54c2a25.png)

修改为

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728028391464-15a33693-ee12-487e-a353-668add68c951.png)

也就是换了激活函数，将ReLU换成了LeakyReLU，

后者参考[https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728028620844-f6ad1780-53a5-453d-ae9e-81171a391e9d.png)

实际上ReLU是max（0，x），而LeakyReLU则是在负分段处换成了另外一个函数

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728028598866-470060c6-07bb-44c2-9baa-7459af49c8a7.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728028841877-8a4a2898-eb05-489b-b57c-d495ae7429c1.png)

如何理解dead neuron

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728029825680-acdd7642-0f0c-4e8d-bcc0-8eec331c2424.png)![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728029878032-96ddafba-d43f-4c88-b3b5-b13e10f0e7d4.png)![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728029892787-bb7169d5-dcee-4bfc-9ddf-daf2e05664fa.png)![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728029913312-8d9bb0d2-5df1-444e-b8b7-201421be15ef.png)![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728029925233-4fc258d4-f9b6-4e42-956a-ab6cfa890a88.png)

**todo：理解dead neuron以及ReLU训练的缺点**

另外的网络优化器中使用了SGD之外的其他算法，参考提示

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728030286085-65d5df27-e3f7-4b9c-8d19-0029a5919202.png)[https://pytorch.org/docs/stable/optim.html](https://pytorch.org/docs/stable/optim.html)，提供了一些优化算法

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728030567072-622469e1-739d-44c9-9317-fdd20252c541.png)

该strong示例中将SGD修改为了Adam算法，

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728030753857-d29f6e38-1dd3-46b9-8929-e2f81b6e6e7d.png)

（至于learning rate还是那句话，可以直接在config中修改，所以不必写成*10这种）

此处暂时不修改learning rate：

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728031161911-5f19da98-268b-4bcd-a259-4a35001a075e.png)  
可能是该参考例子中认为strong baseline的lr太低，导致参数更新慢，所以修改了x10

如果不修改的话那门效果如下：

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728032054716-ab3b5ecc-b1fd-4ef4-b359-688483374045.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728032240133-6734085f-0a28-4f17-a2ed-01506f6ea36e.png)

可以看到loss在后面基本上没有变化，

修改了之后

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728034522687-7f6a64bf-80a5-486e-ac28-386790b53191.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728033510397-36a42af7-ca7d-4076-9624-7a79b73d34f0.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728034153988-1ee50513-30d7-44ee-b239-3dddc8553cdc.png)

实际上处理之后蓝色线条是现在的，墨黑色线条是之前的，

可以看出收敛速度快了很多，但是后面又变平了

总而言之，效果勉勉强强

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728034398163-9b9a1b50-b895-4f15-b471-9f3cfe6597d4.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728034423148-09222f62-21da-4645-97e0-d0c9323aea38.png)

然后就是train以及test集上的数据应该更关注哪个：

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728034456772-52fb9050-143b-456f-9645-95366d510a0d.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728034473375-0896c2ba-f8f3-4df7-830c-3a46f9380962.png)

所以此处我对代码进行了一些可视化的修改，主要是在tensorboard中的loss的统一可视化；

在Training Loop阶段：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728037162498-86977c30-05a0-4e58-bdf2-7478547a79c7.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728037179045-f93b4692-8934-47a0-891b-139777eed5cd.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728037204812-396d23dc-9db9-45f9-bc4b-bb722b97041d.png)

其实主要是将原来分散的writer.add_scalar现在整合在了一起（原本分别计算train以及valid，现在合并在一起记录，提供词典dict）

（3）boss baseline：

首先是修改网络架构：

从strong的  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728038834169-7a46a894-c74a-4614-93e8-3af785c3b324.png)

修改为：

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728040162945-64103838-83bc-4261-8a2f-6e6edd6881e0.png)

主要修改内容解读如下：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728039341604-3c37450f-bc4f-4cf3-a847-09984bb421ed.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728039385558-4498255a-4330-4cba-a9c1-16c3b0d55a31.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728039404392-d7c38a13-88d3-40ef-ab2e-14d0359eab9c.png)

即修改后的：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728040183689-62c507ba-4390-4ffa-86d2-f62b98f080d2.png)

**网络架构的修改包括网络包括哪些层，以及每一层中的网络neuron的数目**

也就是深度deep以及宽度fat方向的修改，前者修改什么层连接什么层，后者修改每一层中神经元的数目等。



前者在网络架构上主要是加入了batchNorm以及dropout操作，也就是这两层（其余的全连接层、激活层等都没有修改） ，所谓的BN层以及Dropout层

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728039914633-5b51377c-9444-43c0-8496-b9502455e3d1.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728039894435-f61fa350-f59f-40ad-9f96-e2a855ee7d9b.png)

可以参考[https://blog.csdn.net/m0_63007797/article/details/128742638](https://blog.csdn.net/m0_63007797/article/details/128742638)

其中batchnorm1d函数可以参考[https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)

**当然还有一个问题就是什么时候使用这些层：什么时候使用批量归一化（Batch Normalization）和Dropout层操作？**

暂时借助gpt的参考作为思考方向，内容正误未知：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728041685346-72bc9b94-f109-4061-b05d-2cfe15b532ca.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728041699982-8faa3c3f-b849-4fba-90fc-05c61d12cff2.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728041722909-2a15af77-6172-4814-ba27-470cf9b431f1.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728041738415-00bb5b7d-f65b-401e-828b-730df5f80098.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728041748850-897b15d7-938b-407d-aa77-d55e4d54ec2e.png)

**所以该示例中的boss baseline也仅仅作为参考，至于batchNorm以及Dropout层要添加在网络的什么地方另外考虑**

总之我们目前修改了网络的架构，在deep方向上增加了BN以及Dropout两层网络（至于什么地方/时候要加，分别加多少层暂定），**至少我们从经验上能够学习到要优化NN可以加这两层。

另外修改的部分是feature select部分：主要是使用ML方法选feature

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728043005810-a671d04e-f20e-49f2-87a8-45645f64558f.png)

主要解读：

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728043886883-04d46560-e401-4114-b939-cb7853929ef1.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728043901007-f0e7226d-cd4d-44eb-a032-2e5209e3940c.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728043913129-3bc2a3c2-0f93-4492-9967-8261ce9a4f2b.png)



可以参考[https://github.com/Hoper-J/HUNG-YI_LEE_Machine-Learning_Homework/tree/master/HW01](https://github.com/Hoper-J/HUNG-YI_LEE_Machine-Learning_Homework/tree/master/HW01)

同样使用机器学习ML的方法来选取feature，同样是使用sklearn.feature_selection.SelectKBest 来进行特征选择

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728041547357-f0edfdaf-a5ea-4458-bcf6-44c65b64498f.png)

另外参考：[https://github.com/sotaBrewer824/LHY_MLDL/blob/main/hw01/hw01.ipynb](https://github.com/sotaBrewer824/LHY_MLDL/blob/main/hw01/hw01.ipynb)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728044276355-a2c5461c-ddab-4932-8874-98f9296f431d.png)

实际基本上完全是boss baseline中的思路以及代码，只不过修改了超参数k=17个feature

**k这一部分我们稍后可以自行修改，可以依据自己的理解来选取使用哪些feature**

前面的feature修改了，那么后面select_feat函数也要修改：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728044452370-36dd3ee8-d4b3-45b2-a3a8-c64d354387b0.png)  


然后就是对优化器（优化算法）的修改，还是采用Adam优化器，但是修改了参数，以及对于learning rate的参数也进行了设置

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728044989372-e6661f95-4afa-4650-8725-1f41dd455b8c.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728046693003-9ae52521-30b8-4aae-befb-ec1cfb949c99.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728046739378-5fd489e3-b5f0-4622-9434-21bc6b8e9d4e.png)![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728046750287-4e7b2e63-b653-412c-871e-57f7e6cf8717.png)

如果使用了scheduler来调整学习率，需要在适当时机调用

一般是在train的每个阶段即每个epoch训练之后使用scheduler

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728047719657-c07103cc-68f2-45c4-b653-823fd5690c08.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728047776031-3c68770e-ef3d-477a-9e92-cf4b93fda0db.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728047803357-c3f23890-74fa-46d6-b9a5-57223b8c8953.png)

参考资料中是使用基于epoch的学习率调整，所以在参考例子中

是在epoch处理之后、valid之前调用

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728047900382-e04701b2-f18d-457a-ae30-a0d99206a4cc.png)

另外参考[https://github.com/sotaBrewer824/LHY_MLDL/blob/main/hw01/hw01.ipynb](https://github.com/sotaBrewer824/LHY_MLDL/blob/main/hw01/hw01.ipynb)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728047986287-61b64ad1-6cfc-4214-a778-3f5ccd4fe54d.png)

本质上也是在train的epoch之后、valid之前（其实与loss的记录没有顺序关系影响）  




其余修改的地方影响也不是很大：

主要是训练epoch数目的延长，以及learning rate的修改（可以在前面optimizer中修改）

以及修改了一个early stop

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728048272267-f83db31c-f8ae-4461-8783-71b8b67564e5.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728048298059-a7dd82f0-85e5-421b-912d-8d2f32a78c50.png)

参考资料中对于early stop进行的操作与原来strong baseline中的没有区别

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728048515160-18cc3eef-013a-4705-b0b3-fe7af6605c43.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728048531435-c9f8ff7f-2dbe-45e1-8bf9-85cd9d35771a.png)

本质上都是超过early stop数目之后会执行终止（break或者return）

另外参考资料中使用了train_losses，valid_losses来记录训练以及valid过程中的loss，所以后面绘制loss曲线的时候没有使用tensorboard，而是自己使用记录的数据用matplotlib进行了绘制；

这一部分此处我不做改动，还是使用原来的tensorboard。



另外参考部分还对输入数据进行了修改：

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728049157469-bf685787-8b68-486f-8108-a7a9a6c9cbd1.png)

其实主要是norm归一化，和前面的batchnorm类似

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728049080894-beff6a71-191d-437d-a60c-3d3c0fc3636e.png)

此处不执行。

实际执行过程中：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728049726233-57e23c8b-7927-4396-bb45-b3b57e05d8b7.png)

很显然之前选择的前4天的阳性率也是高相关的feature，排在top4。

实际执行过程：  
![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728050668758-10da0854-48d1-4802-8d3d-5d82b4daceca.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728050712685-031ac4bf-c741-4a0b-af0e-1718d33a580e.png)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728051818905-e779815b-0e61-475d-a12c-17312289848c.png)

曲线是合并在了一起，可以方便的查看

valid在train下面，loss也还行

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728034473375-0896c2ba-f8f3-4df7-830c-3a46f9380962.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0)

6，所有代码都上传到github中：

[https://github.com/MaybeBio/Deep_learning/tree/main/Li_Hong_Yi_2021_2022/HW1](https://github.com/MaybeBio/Deep_learning/tree/main/Li_Hong_Yi_2021_2022/HW1)

![](https://cdn.nlark.com/yuque/0/2024/png/33753661/1728056266112-5d206213-ab04-45b0-8dbd-e152e0771844.png)

只保留：  
**官网上的脚本（也就是simple baseline）是*_original

官网参考过的脚本是*_official，自己注释过的是*_annotated（当然都是simple baseline）

然后参考github大佬复现学习的脚本就按照*_medium、*_strong、*_boss等等级来记录**

