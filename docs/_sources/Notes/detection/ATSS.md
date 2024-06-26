# ATSS

**标题：** Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection

**作者：** Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, Stan Z. Li

**机构：** CBSR, NLPR, CASIA; SAI, UCAS; AIR, CAS; BUPT; Westlake University

**摘要：**
这篇论文首先指出，锚点（anchor-based）检测器和无锚点（anchor-free）检测器之间的根本区别在于它们如何定义正例和负例训练样本。作者提出，如果在训练过程中采用相同的正负样本定义，则无论最终是从一个框还是一个点回归，它们之间的性能差异并不明显。基于这一发现，作者提出了一种自适应训练样本选择（ATSS）方法，该方法根据对象的统计特性自动选择正负样本，显著提高了锚点检测器和无锚点检测器的性能，并弥合了它们之间的差距。此外，论文还讨论了在图像上为检测对象而平铺多个锚点的必要性。在MS COCO数据集上的广泛实验支持了上述分析和结论。通过引入ATSS，作者在不引入任何开销的情况下大幅度提高了最先进检测器的性能，达到了50.7% AP。代码已开源。

**1. 工作内容与动机：**
工作内容是提出ATSS方法，自动根据对象的统计特性选择正负样本，以提高检测器性能。动机是缩小锚点检测器和无锚点检测器之间的性能差距，并探索如何更有效地选择训练样本。

**2. 解决的问题：**
解决的问题是如何在训练对象检测器时更有效地定义正例和负例样本。

**3. 新问题：**
这不是一个全新的问题，但ATSS提供了一种新的解决方案。

**4. 科学假设：**
假设是不同的正负样本选择策略对于训练对象检测器的性能有显著影响。

**5. 相关研究：**
相关研究包括锚点检测器（如Faster R-CNN, SSD等）和无锚点检测器（如CornerNet, CenterNet等）。这些研究可以根据它们是基于锚点还是无锚点的方法来分类。领域内值得关注的研究员包括Tsung-Yi Lin, Kaiming He, Jifeng Dai等。

**6. 解决方案关键：**
解决方案的关键是ATSS，它通过统计特性来自动选择正负样本，几乎不需要任何超参数。

**7. 实验设计：**
实验设计包括在MS COCO数据集上进行训练和测试，使用ResNet-50作为骨干网络，并比较了不同检测器的性能。

**8. 数据集与代码：**
使用的是MS COCO数据集。代码已在GitHub上开源。

**9. 实验结果：**
实验结果表明，ATSS显著提高了锚点检测器和无锚点检测器的性能，支持了论文的科学假设。

**10. 论文贡献：**
贡献包括指出正负样本选择对于对象检测器性能的重要性，并提出了ATSS方法来自动选择正负样本，从而提高了检测器性能，并在COCO数据集上达到了最先进的性能。

**11. 下一步工作：**
下一步工作可以包括进一步探索和改进正负样本选择策略，或者将ATSS方法应用于其他类型的检测任务或数据集。

回答问题

1. **这篇论文做了什么工作，它的动机是什么？**
   论文提出了ATSS方法，自动根据对象的统计特性选择正负样本，以提高检测器性能。动机是缩小锚点检测器和无锚点检测器之间的性能差距。

2. **这篇论文试图解决什么问题？**
   论文试图解决训练对象检测器时如何更有效地定义正例和负例样本的问题。

3. **这是否是一个新的问题？**
   这不是一个全新的问题，但提供了一种新的解决方案。

4. **这篇文章要验证一个什么科学假设？**
   假设是不同的正负样本选择策略对于训练对象检测器的性能有显著影响。

5. **有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？**
   相关研究包括锚点检测器和无锚点检测器。可以根据它们是基于锚点还是无锚点的方法来分类。领域内值得关注的研究员包括Tsung-Yi Lin, Kaiming He, Jifeng Dai等。

6. **论文中提到的解决方案之关键是什么？**
   解决方案的关键是ATSS，它通过统计特性来自动选择正负样本，几乎不需要任何超参数。

7. **论文中的实验是如何设计的？**
   实验设计包括在MS COCO数据集上进行训练和测试，使用ResNet-50作为骨干网络，并比较了不同检测器的性能。

8. **用于定量评估的数据集上什么？代码有没有开源？**
   使用的是MS COCO数据集。代码已在GitHub上开源。

9. **论文中的实验及结果有没有很好地支持需要验证的科学假设？**
   是的，实验结果表明ATSS显著提高了锚点检测器和无锚点检测器的性能，支持了论文的科学假设。

10. **这篇论文到底有什么贡献？**
    贡献包括指出正负样本选择对于对象检测器性能的重要性，并提出了ATSS方法来自动选择正负样本，从而提高了检测器性能，并在COCO数据集上达到了最先进的性能。

11. **下一步呢？有什么工作可以继续深入？**
    下一步工作可以包括进一步探索和改进正负样本选择策略，或者将ATSS方法应用于其他类型的检测任务或数据集。
