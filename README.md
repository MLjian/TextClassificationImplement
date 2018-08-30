# 1 简介
['达观杯'文本智能处理挑战赛官网](http://www.dcjingsai.com/common/cmpt/“达观杯”文本智能处理挑战赛_竞赛信息.html)<br>
&#8195;该库用于'达观杯'比赛的文本分类任务的基础实现，主要包括机器学习(ml)和深度学习(dl)两大部分，机器学习部分基于sklearn/lightgbm包实现，深度学习部使用pytorch深度学习框架。其中，机器学习部分主要包含特征工程和分类器两大部分，特征工程部分主要针对文本分类任务的 hash/lsa/lda/doc2vec特征提取/特征选择/特征组合/特征构造进行了实现，而分类器部分主要有逻辑回归/SVM/随机森林/Bagging/Adaboost/GBDT/Xgboost/LightGBM等。深度学习主要实现了word2vec/构建lstm模型/训练可视化等。（注：此库只是基础实现，并不是最优！！！）<br>
# 2 ml（机器学习）
- 1）**运行环境**<br>
sklearn/xgboost/lightgbm<br>
- 2）**文件夹说明**<br>
[data]:用于存放原始数据集。([数据集下载链接](https://pan.baidu.com/s/17UjEEcB2taT_HvU1FC1bCQ))<br>
[features]:用于存放特征工程的代码以及生成的特征文件。<br>
[code]:用于存放训练的代码。<br>
[results]:用于存放测试集的训练结果，以便提交给比赛官方。<br>
[for beginner]：用较为简单的代码格式帮助代码熟练度不够的同学入门，涉及自动调参。(选看)<br>
- 3）**使用案例**<br>
（1）生成tfidf特征<br>
直接运行features文件夹中的**tfidf.py**;<br>
（2）生成lsa特征<br>
运行features文件夹中的**lsa.py**；(注：运行前请确保tfidf特征已经生成)<br>
（3）生成lda特征<br>
运行features文件夹中的**lda.py**；(注：运行前请确保tf特征已经生成)<br>
（4）使用逻辑回归分类器进行训练<br>
修改code文件夹中的**sklearn_config.py**中的*clf_name='lr'*,然后运行**sklearn_train.py**即开始训练。<br>
（5）使用lgb进行训练<br>
修改code文件夹中的**lgb.py**中的*features_path*，然后直接运行**lgb.py**即可开始训练。<br>
- 4）**提高模型分数关键**<br>
（1）特征工程：<br>
做更多更好的特征，然后进行融合，形成新的特征，正常来讲每增加一些有用的特征，模型就会提升一些；<br>
对于article的使用，将article进行和word_seg一样的特征抽取，然后合并到word_seg特征中；<br>
（2）集成学习：<br>
多个好而不同的单模型进行融合，就是将各个模型结果进行投票；<br>
- 5）**比赛新baseline**<br>
用ensemble_spar.py形成的特征 + LinearSVC --> 0.778多(只是简单随便跑的，还未进行调优)<br>
# 3 dl（深度学习）
- 1）**运行环境**<br>
pytorch/visdom
- 2）**文件夹说明**<br>
[n_pad]：不对句子进行截断或补零。<br>
[pad]：对句子进行截断或补零，以保证输入神经网络里的每条句子长度一样。<br>
  [data]:用于存放原始数据集和处理后的数据集。([数据集下载链接](https://pan.baidu.com/s/17UjEEcB2taT_HvU1FC1bCQ))<br>
  [models]：用于存放网络结构的文件。<br>
  [word2vec]：用于存放训练词向量的代码以及训练词向量的生成文件。<br>
  [trained_models]：用于存放训练好的网络模型。<br>
  [实验数据]：用于存放实验记录。<br>
  [config.py]：用于对网络结构的参数进行设置。<br>
  [train.py]：用于对网络的训练
- 3）**使用流程**<br>
（1）先将原始数据集下载至data/data_ori；<br>
（2）运行word2vec/train_word2vec.py训练词向量；<br>
（3）运行data/data_process.py ,对数据进行预处理；<br>
（4）配置config文件进行参数的配置，并保存；<br>
（5）运行train.py进行训练；<br>
- 4）**训练过程**<br>
<div align=center><img width="584" height="375" src="https://github.com/MLjian/TextClassificationImplement/blob/master/dl/n_pad/实验数据/loss.png"/></div><br>
<div align=center>训练集loss
<div align=center><img width="584" height="375" src="https://github.com/MLjian/TextClassificationImplement/blob/master/dl/n_pad/实验数据/acc.png"/></div>
<div align=center>验证集准确率


