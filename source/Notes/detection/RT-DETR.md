# RT-DETR

**标题：** DETRs Beat YOLOs on Real-time Object Detection

**作者：** Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu

**机构：** Baidu Inc, Beijing, China; School of Electronic and Computer Engineering, Peking University, Shenzhen, China

**摘要：** 本文提出了一种名为Real-Time DEtection TRansformer (RT-DETR)的新型实时目标检测器。RT-DETR是首个实时端到端目标检测器，它通过设计高效的混合编码器和最小化不确定性的查询选择机制，显著提高了检测速度和准确性。此外，RT-DETR支持通过调整解码器层数来灵活调整速度，而无需重新训练。在COCO数据集上，RT-DETR-R50和RT-DETR-R101分别达到了53.1%和54.3%的AP，以及108 FPS和74 FPS的检测速度，超越了以往先进的YOLO检测器。

**1. 工作内容与动机：** 动机：YOLO系列模型在实时目标检测中受到非极大值抑制（NMS）的负面影响，导致速度和准确性下降。 工作：提出了RT-DETR，一种无需NMS的实时端到端目标检测器，通过混合编码器和查询选择机制提高速度和准确性。

**2. 解决的问题：** 解决了YOLO系列模型中NMS导致的速度和准确性问题，并提出了一种无需NMS的实时目标检测方法。

**3. 新问题：** 是的，提出了一个新的问题解决方案，即在实时目标检测领域中消除NMS的影响。

**4. 科学假设：** 假设通过改进DETR的编码器结构和查询选择机制，可以构建一个既快速又准确的实时目标检测器，超越现有的YOLO模型。

**5. 相关研究：**

- 实时目标检测器：YOLO系列。
- 端到端目标检测器：DETR及其变种。
- 领域内值得关注的研究员：Nicolas Carion（DETR的提出者）。 相关研究归类为基于CNN的实时检测器和基于Transformer的端到端检测器。

**6. 解决方案的关键：**

- 高效的混合编码器：通过解耦内部尺度交互和跨尺度融合来提高处理多尺度特征的速度。
- 最小化不确定性的查询选择：为解码器提供高质量的初始查询，以提高准确性。

**7. 实验设计：** 在COCO val2017数据集上进行训练和验证，使用标准的COCO评估指标，包括AP、AP50、AP75以及不同尺度的AP（APS、APM、APL）。

**8. 数据集与代码：** 使用COCO数据集进行定量评估。项目页面提供了更多信息，但文中未明确指出代码是否开源。

**9. 实验结果：** 实验结果表明，RT-DETR在速度和准确性上均超越了先前的YOLO检测器，支持了提出的科学假设。

**10. 论文贡献：**

- 提出了首个实时端到端目标检测器RT-DETR，它在速度和准确性上均超越了YOLO检测器。
- 引入了高效的混合编码器和最小化不确定性的查询选择机制。
- 支持灵活的速度调整，无需重新训练即可适应不同场景。

**11. 下一步工作：**

- 改进对小目标的检测性能。
- 探索使用预训练的大型DETR模型来提升RT-DETR的性能。
- 将RT-DETR应用于更多的实时检测场景，并进行实际部署。
