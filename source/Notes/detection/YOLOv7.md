# YOLOv7

**标题：** YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

**作者：** Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao (Institute of Information Science, Academia Sinica, Taiwan)

**摘要：**
YOLOv7在实时目标检测领域中超越了所有已知的目标检测器，无论是在速度还是准确性方面。在GPU V100上，YOLOv7-E6目标检测器（56 FPS, 55.9% AP）在速度上比基于变换器的检测器SWINL Cascade-Mask R-CNN（9.2 FPS, 53.9% AP）快509%，准确性上高出2%，比基于卷积的检测器ConvNeXt-XL Cascade-Mask R-CNN（8.6 FPS, 55.2% AP）快551%，准确性上高出0.7%。

**1. 问题：**
论文试图解决实时目标检测中速度与准确性之间的权衡问题，提高目标检测器在不同设备上的运行效率和准确性。

**2. 新问题：**
这不是一个全新的问题，但是YOLOv7提出了新的解决方案，以实现更好的速度和准确性权衡。

**3. 科学假设：**
假设通过优化训练过程和模型架构，可以实现不增加推理成本的情况下提高目标检测的准确性。

**4. 相关研究：**
- 实时目标检测器：YOLO系列、FCOS、SSD、RetinaNet等。
- 模型重参数化和动态标签分配：与网络训练和目标检测相关的方法。
- 领域内值得关注的研究员包括但不限于：Joseph Redmon、Ali Farhadi（YOLO系列的主要贡献者）。

**5. 解决方案关键：**
- 设计了可训练的“免费功能包”（trainable bag-of-freebies），包括新的优化模块和方法。
- 提出了“计划重参数化模型”（planned re-parameterized model）和“粗到细引导标签分配”（coarse-to-fine lead guided label assignment）方法。

**6. 实验设计：**
实验使用Microsoft COCO数据集进行，所有模型从头开始训练，不使用预训练模型。设计了针对边缘GPU、常规GPU和云GPU的基本模型，并使用提出的复合缩放方法对模型进行扩展。

**7. 数据集与代码：**
使用的数据集是COCO，源代码已在GitHub上开源：https://github.com/WongKinYiu/yolov7。

**8. 实验结果：**
实验结果表明，YOLOv7在不同模型尺寸上均取得了优异的性能，支持了论文提出的科学假设。

**9. 贡献：**
- 提出了YOLOv7，一个高性能的实时目标检测模型。
- 引入了新的训练方法和模型优化技术，提高了目标检测的准确性和速度。
- 提供了不同规模的模型以适应不同的计算设备。

**10. 下一步工作：**
- 进一步探索和优化YOLOv7模型，以提高对小目标的检测能力。
- 在更多的数据集上测试YOLOv7的性能。
- 探索YOLOv7在不同应用场景中的实用性和效果。

回答问题

1. **问题：** 提高实时目标检测的速度和准确性。
2. **新问题：** 不是新问题，但提出了新的解决方案。
3. **科学假设：** 通过训练过程和模型架构的优化，可以在不增加推理成本的情况下提高检测准确性。
4. **相关研究：** YOLO系列、FCOS、SSD、RetinaNet等。
5. **解决方案关键：** 可训练的“免费功能包”，计划重参数化模型，粗到细引导标签分配。
6. **实验设计：** 在COCO数据集上进行，设计了不同规模的模型。
7. **数据集与代码：** 使用COCO数据集，代码已开源。
8. **实验结果：** 支持假设，YOLOv7在不同模型尺寸上均取得了优异性能。
9. **贡献：** 提出了YOLOv7和一系列新的训练和模型优化技术。
10. **下一步工作：** 提高小目标检测能力，测试更多数据集，探索应用场景。