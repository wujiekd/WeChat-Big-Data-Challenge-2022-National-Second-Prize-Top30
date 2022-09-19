## 代码说明

### 环境配置

所需环境在 `requirements.txt` 中定义。

### 数据

* 使用大赛提供的有标注数据（10万）和无标注数据（大约100万）。
* 未使用任何额外数据。

### 预训练模型

* 使用了 huggingface 上提供的 `hfl/chinese-macbert-base` 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base

* 使用了 huggingface 上提供的 `hfl/chinese-roberta-wwm-ext` 模型。链接为： https://huggingface.co/hfl/chinese-roberta-wwm-ext

* 使用了 微软开源的swin-transformer预训练模型。 opensource_models/swin_base_patch4_window7_224_22k.pth

### 算法描述

* 文本处理使用的长度为260，通过对三类文本数据进行数据探索性分析，根据百分位数确定，title 长度为98 ， asr为90，ocr为68.。对三个序列分别tokenizer后再进行合并，再添加CLS和SEQ特定符。

* 图像处理为间隔采样16帧

* 采用了两类模型进行集成，参考了以下经典架构设计模型
  * 单流模型 VL-Bert架构 
  * 双流模型 ALBEF架构 

* 单流模型和双流模型在9：1的SKF验证中，在不需要预训练的前提下，线上A榜均可达到0.685左右的分数

* 预训练采用了MFM，ITM，MLM

* 主要模型思路（更有效且更快的模型融合）
1. 使用100w无标签数据对单流模型进行预训练（图像提取模型固定，提前采用swin_base_patch4_window7_224_22k.pth进行抽帧）；
2. 使用10w有标签数据对单流模型进行微调（图像提取模型放开）；
3. 加载单流模型训练好的swin_base部分的权重，抽取图像帧；
4. 使用10w有标签数据对双流模型进行微调（图像提取模型固定，第3步已抽取特征）；
5. 加载单流和双流模型训练权重，在推理时，以softmax输出进行0.55 ： 0.45 的权重进行融合。

注：单流模型和双流模型全量数据训练的最佳step通过9：1的SKF划分训练得到的结果进行确定最佳step训练的模型

模型优点：效率高。预训练只训练一次，且冻结了图像提取的backbone，使得预训练（包含抽帧）仅需24小时内即可完成，然后单流模型的训练需要12个小时，而双流模型（包含抽帧）仅需要1.5个小时、

* 采用APEX的FP16进行压缩模型，提升了一倍的效率(预训练不采用FP16，防止NAN），另外TensorRT也有实现，但我们的模型不够多，并没有采用更复杂的TensorRT进行推理

* 其他Trick仅采用了label smoothing。



### 分工合作
1. 单流模型的设计：白竟宏，张钟瑾 
2. 双流模型的设计：张钟瑾，卢科达
3. 图像backbone的选取： 卢科达
4. 模型融合：白竟宏
5. fp16以及tensorRT：张钟瑾
6. 标签平滑：白竟宏
7. 流程设计：白竟宏，张钟瑾，卢科达

### 性能


B榜融合测试性能：**0.705623**



### 训练流程

* sh init.sh

* sh train.sh
  

### 测试流程

* sh inference.sh即可
