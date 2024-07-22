# YOLOS

**标题**: You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection

**作者**: Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu

**机构**: 华中科技大学, Horizon Robotics

**摘要**: 本文提出了You Only Look at One Sequence (YOLOS)，这是一系列基于纯Transformer架构的对象检测模型，目标是探索Transformer在2D对象和区域级别识别任务中的潜力，同时尽可能少地修改原始架构和引入目标任务的归纳偏差。

**1. 论文试图解决的问题**:
论文探讨了Transformer模型是否能够从纯序列到序列的角度出发，以最小的2D空间结构知识，执行2D对象和区域级别的识别任务。

**2. 是否是一个新的问题**:
是的，这个问题是新的。尽管Transformer在自然语言处理（NLP）中已经非常成功，但在计算机视觉（CV）中，特别是在对象检测这样的复杂任务上，直接应用Transformer仍然是一个相对较新和具有挑战性的领域。

**3. 文章要验证的科学假设**:
假设是：预训练的Transformer能够成功地从图像识别任务迁移到更为复杂的2D对象检测任务。

**4. 相关研究**:
- **ViT-FRCNN**: 使用预训练的ViT作为特征提取器的Faster R-CNN对象检测器。
- **DEtection TRansformer (DETR)**: 使用Transformer编码和解码CNN特征进行对象检测。
- **CNN与Transformer结合的研究**: 将CNN和自注意力机制结合起来以提高对象检测性能。
- **领域内值得关注的研究员**: 本文没有特别指出，但提到了多个与Transformer和对象检测相关的研究工作，如Dosovitskiy, Carion等。

**5. 解决方案的关键**:
YOLOS的关键是在ViT的基础上进行最小的修改，用100个[DET]标记替换ViT中的[CLS]标记，并使用二分图匹配损失来进行对象检测，避免了将ViT输出序列重新解释为2D特征图。

**6. 实验设计**:
实验包括在ImageNet-1k数据集上预训练，然后在COCO对象检测基准上进行微调。作者还研究了不同的预训练策略（有监督和自监督）对迁移到COCO的影响。

**7. 数据集和代码开源**:
使用的数据集是ImageNet-1k和COCO。代码和预训练模型已在GitHub上开源，地址为：https://github.com/hustvl/YOLOS。

**8. 实验结果支持假设**:
实验结果表明，YOLOS在COCO对象检测基准上取得了竞争性的性能，证明了预训练的Transformer能够有效迁移到对象检测任务。

**9. 论文贡献**:
- 提出了YOLOS，一个基于最少修改的ViT架构的对象检测模型。
- 证明了2D对象检测可以以纯序列到序列的方式完成。
- 展示了预训练Transformer在对象检测任务中的通用性和迁移能力。
- 提供了一个用于评估不同预训练策略的挑战性基准。

**10. 下一步工作**:
- 进一步改进YOLOS的性能，使其达到或超过当前最先进的对象检测模型。
- 探索自监督预训练在对象检测中的潜力。
- 研究如何更高效地将预训练的Transformer模型适应到下游视觉任务中，减少迁移学习的计算成本。

回答问题

1. **论文试图解决的问题**: 探索Transformer模型在2D对象和区域级别识别任务中的应用潜力。

2. **是否是一个新的问题**: 是的，这是一个新的问题。

3. **文章要验证的科学假设**: 预训练的Transformer能够有效迁移到2D对象检测任务。

4. **相关研究**: ViT-FRCNN, DETR, CNN与Transformer结合的研究。值得关注的研究员包括Dosovitskiy, Carion等。

5. **解决方案的关键**: YOLOS在ViT基础上进行了最小的修改，使用[DET]标记和二分图匹配损失。

6. **实验设计**: 在ImageNet-1k上预训练，然后在COCO上微调，研究不同预训练策略的影响。

7. **数据集和代码开源**: 使用了ImageNet-1k和COCO数据集，代码已开源。

8. **实验结果支持假设**: 是的，实验结果支持了假设。

9. **论文贡献**: 提出YOLOS模型，证明了Transformer的迁移能力和通用性，提供了评估预训练策略的基准。

10. **下一步工作**: 改进YOLOS性能，探索自监督预训练，研究高效迁移学习策略。

---
