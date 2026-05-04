# 医学影像 AI 及 CNN 肺炎分类：核心方法与文献综述

> 按核心程度排序，罗列关键文献与内容要点，标注完整出处和段落位置。
> 用于课程论文「文献综述」章节撰写。

---

## 一、核心直接文献（Direct Methods — 肺炎 X 光分类）

### [L1] Rajpurkar et al., 2017 — CheXNet（核心必引）

**完整出处** Rajpurkar, P., Irvin, J., Zhu, K., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M.P. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. arXiv:1711.05225.

**关键内容**

- 使用 DenseNet-121 架构，在 ChestX-ray14 数据集（112,120 张胸片，来自 30,805 名患者）上训练。
- 模型在肺炎分类任务上 **AUC = 0.76**，超过放射科医生平均水平（AUC = 0.73）。
- 使用 **Label smoothing** 减少过拟合。
- 14 种病理同时预测（multi-label），肺炎仅为其中之一。
- 证明了深度学习在胸片诊断上 **可以超越普通放射科医生**，是自 AlexNet 以来医学影像 AI 的里程碑式工作。

**论文位置** 摘要段 + 结果段（第 1–3 页）。

---

### [L2] Kermany et al., 2018 — ChestXRay2017 数据集原论文（核心数据集）

**完整出处** Kermany, D.S., Goldbaum, M., Cai, W., Valentim, C.C.S., Liang, H., Baxter, S.L., Liu, J., Nie, D., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122–1131.

**关键内容**

- 发布了 **ChestXRay2017** 数据集：5,856 张儿童胸片，分为 Normal / Bacterial Pneumonia / Viral Pneumonia 三类。
- 使用 **Google Inception V3** 迁移学习，在二分类（Normal vs. Pneumonia）任务上达到 **Accuracy = 92.8%**，AUC = 0.97。
- 引入 **OCT 图像** 并行研究（眼部疾病），证明方法论可迁移。
- 强调 **可解释性**：使用 Grad-CAM 可视化，模型关注区域与医生标注区域高度一致。
- 数据集获取：Kaggle（`paultimothmooney/chest-xray-pneumonia`），可直接用于本课程论文。

**论文位置** 结果段（第 3–5 页，Figure 2 及 Table 1）。

---

### [L3] Stephen et al., 2019 — CNN 架构对比（方法参考）

**完整出处** Stephen, O., Sain, M., Maduh, U.J., Jeong, D.U. (2019). *An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare*. Journal of Healthcare Engineering, Vol. 2019, Article ID 4180949.

**关键内容**

- 系统对比了 **5 种 CNN 架构**：VGG16、VGG19、Inception V3、ResNet-50、DenseNet-201。
- 在 ChestXRay2017 数据集上，**DenseNet-201 表现最佳**，Accuracy = 95.3%（优于原始 Kermany 的 Inception V3）。
- 证明 **迁移学习 + 数据增强**（rotation、shift、flip）是提升胸片分类性能的关键。
- 提供了详细的 **超参数设置**（batch size、learning rate、optimizer 选择），可直接参考用于本课程论文实验设计。

**论文位置** 实验部分（第 4–6 页，Table 3：各架构性能对比）。

---

## 二、可解释性方法（Interpretability — Grad-CAM 系列）

### [L4] Selvaraju et al., 2017 — Grad-CAM 原论文（核心方法）

**完整出处** Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 618–626.

**关键内容**

- 提出 **Grad-CAM**：利用目标类别对最后一层卷积特征图的梯度，生成 **类别判别性定位图**（class-discriminative localization map）。
- 不需要修改网络结构，**适用于任何 CNN 架构**（VGG、ResNet、DenseNet 均适用）。
- 输出为 **热力图（heatmap）**，可叠加在原始图像上展示模型「看到」的区域。
- 在 ImageNet 分类、图像描述（captioning）、VQA 任务上均验证了有效性。
- **医学影像应用价值**：帮助医生理解模型决策依据，满足临床落地对「可解释 AI」（XAI）的要求。

**论文位置** 方法段（第 2–3 页，公式 1–2：Grad-CAM 计算流程）。

---

### [L5] Chattopadhyay et al., 2018 — Grad-CAM++（改进版）

**完整出处** Chattopadhyay, A., Sarkar, A., Howlader, P., & Balasubramanian, V.N. (2018). *Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks*. IEEE Winter Conference on Applications of Computer Vision (WACV), 839–847.

**关键内容**

- 改进 Grad-CAM，引入 **高阶梯度信息**（higher-order gradients），使热力图定位更精确。
- 对**同类别多实例**场景（如一张胸片中有多处感染区域）定位效果优于原始 Grad-CAM。
- 可作为本课程论文的 **进阶可解释性方法**（如果时间允许，可将 Grad-CAM 与 Grad-CAM++ 对比）。

**论文位置** 方法段（第 2–3 页）。

---

## 三、网络架构基础（Backbone Architectures）

### [L6] He et al., 2016 — ResNet（本论文基线架构）

**完整出处** He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.

**关键内容**

- 提出 **残差连接（Residual Connection / Skip Connection）**，解决深层网络梯度消失/爆炸问题。
- **ResNet-18** 是本论文选定的基线架构，在 ImageNet 上预训练后做迁移学习。
- 核心公式：`H(x) = F(x) + x`（残差块输出 = 残差函数 + 恒等映射）。
- 实验证明：**网络深度增加不会降低性能**（只要有残差连接），在 ImageNet 上 top-5 error 降至 3.57%。

**论文位置** 方法段（第 3–4 页，Residual Block 结构图） + 实验结果（Table 2）。

---

### [L7] Huang et al., 2017 — DenseNet（ChestXRay2017 最佳架构）

**完整出处** Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K.Q. (2017). *Densely Connected Convolutional Networks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4700–4708.

**关键内容**

- 提出 **DenseNet**：每一层都与之前所有层直接连接（dense connectivity），特征重用效率高。
- 参数效率优于 ResNet（DenseNet-201 参数约为 ResNet-50 的 1/2）。
- 在 ChestXRay2017 上，**DenseNet-201 性能优于 ResNet-18**（Stephen et al., 2019 验证）。
- 可选：如果 ResNet-18 基线性能不理想，可在阶段二换用 DenseNet-121/201。

**论文位置** 方法段（第 2–3 页，Dense Block 结构） + 实验结果（Table 1）。

---

## 四、注意力机制与改进方法（Improvement Methods）

### [L8] Woo et al., 2018 — CBAM（可选改进模块）

**完整出处** Woo, S., Park, J., Lee, J.Y., & Kweon, I.S. (2018). *CBAM: Convolutional Block Attention Module*. Proceedings of the European Conference on Computer Vision (ECCV), 3–19.

**关键内容**

- 提出 **CBAM（Convolutional Block Attention Module）**：串联 **通道注意力（Channel Attention）** 和 **空间注意力（Spatial Attention）** 两个独立模块。
- 可无缝插入任何 CNN 架构（包括 ResNet-18），**计算开销极小**（约增加 <1% 参数量）。
- 在 ImageNet 分类任务上，ResNet-50 + CBAM 达到 **top-1 error = 24.62%**（单独 ResNet-50 为 25.71%）。
- **医学影像应用**：通道注意力帮助模型关注「有诊断价值的特征通道」，空间注意力帮助聚焦「感染区域的空间位置」。

**论文位置** 方法段（第 3–5 页，Channel Attention + Spatial Attention 结构图） + 实验结果（Table 1）。

---

### [L9] Lin et al., 2017 — Focal Loss（处理类别不平衡）

**完整出处** Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2999–3007.

**关键内容**

- 提出 **Focal Loss**：改进标准 Cross Entropy Loss，降低「易分类样本」的权重，使模型专注于「难分类样本」。
- 公式：`FL(p_t) = -α(1 - p_t)^γ log(p_t)`，其中 `γ` 控制难样本权重（通常 γ=2）。
- 在医学影像数据集中，**类别不平衡是普遍问题**（正常样本远多于患病样本），Focal Loss 是标准解决方案之一。
- 可作为本课程论文阶段二的改进方向（替换 Cross Entropy Loss）。

**论文位置** 方法段（第 3–4 页，Focal Loss 公式及原理） + 实验结果（Table 1–2）。

---

## 五、数据增强与正则化（Data Augmentation）

### [L10] Zhang et al., 2018 — MixUp（数据增强进阶）

**完整出处** Zhang, H., Cisse, M., Dauphin, Y.N., & Lopez-Paz, D. (2018). *mixup: Beyond Empirical Risk Minimization*. International Conference on Learning Representations (ICLR).

**关键内容**

- 提出 **MixUp**：训练时将两个样本及其标签按比例线性叠加，生成「虚拟样本」。
- 公式：`x̃ = λx_i + (1-λ)x_j`，`ỹ = λy_i + (1-λ)y_j`，其中 `λ ~ Beta(α, α)`。
- 降低模型对对抗样本的敏感性，提升泛化能力。
- 在医学影像任务中，MixUp 被证明能有效提升小数据集上的性能（ChestXRay2017 仅 5,856 张，适合用 MixUp）。

**论文位置** 方法段（第 2–3 页） + 实验结果（Table 1–2）。

---

### [L11] Yun et al., 2019 — CutMix（数据增强进阶）

**完整出处** Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J., & Yoo, Y. (2019). *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 6023–6032.

**关键内容**

- 提出 **CutMix**：将一张图像的随机矩形区域剪切并粘贴到另一张图像上，标签按区域面积比例混合。
- 比 MixUp 更「空间局部化」，生成的样本更自然，模型学到的特征更具可定位性。
- 在 CIFAR 和 ImageNet 上均超越 MixUp。
- 可作为阶段三的数据增强方案（与 Focal Loss 组合使用）。

**论文位置** 方法段（第 3–4 页） + 实验结果（Table 1–3）。

---

## 六、医学影像 AI 综述类文献（Survey Papers — 用于引言/背景）

### [L12] Litjens et al., 2017 — 医学影像深度学习综述（必引综述）

**完整出处** Litjens, G., Kooi, T., Bejnordi, B.E., Setio, A.A.A., Ciompi, F., Ghafoorian, M., van der Laak, J.A., van Ginneken, B., & Sánchez, C.I. (2017). *A Survey on Deep Learning in Medical Image Analysis*. Medical Image Analysis, 42, 60–88.

**关键内容**

- 全面综述深度学习在医学影像中的应用：**分类、检测、分割、注册** 四大任务。
- 覆盖影像模态：X 光、CT、MRI、超声、病理切片、眼底照相。
- 总结 **迁移学习在医学影像中的必要性**（标注数据少、标注成本高）。
- 指出 **可解释性是医学影像 AI 落地的核心障碍**，需重点研究。

**论文位置** 全文（42 页），重点读第 2–3 节（CNN 方法）和第 6 节（讨论与展望）。

---

### [L13] Wang et al., 2021 — 胸部 X 光 AI 诊断综述

**完整出处** Wang, H., Jia, H., Li, Y., Omote, K., & Emoto, M. (2021). *Deep Learning for Pneumonia Detection: A Review*. IEEE Access, 9, 125105–125124.

**关键内容**

- 专门综述 **深度学习在肺炎 X 光检测中的应用**。
- 对比了 2016–2020 年间 **15+ 篇肺炎 X 光分类论文** 的数据集、架构、性能指标。
- 总结共性结论：① 迁移学习是标准做法；② DenseNet 和 ResNet 是最常用架构；③ 数据增强对小数据集至关重要；④ 可解释性方法是未来研究方向。
- **直接支持本课程论文的文献综述章节**，可大量引用。

**论文位置** 第 2 节（数据集综述） + 第 3 节（方法综述） + Table 2（各论文性能对比表）。

---

## 七、评估指标与实验设计参考

### [L14] Powers, 2011 — 评估指标定义（方法章节引用）

**完整出处** Powers, D.M.W. (2011). *Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation*. Journal of Machine Learning Technologies, 2(1), 37–63.

**关键内容**

- 严格定义 **Accuracy、Precision、Recall、F1-score、ROC、AUC** 等指标的统计学含义。
- 指出在类别不平衡数据集中，**Accuracy 是误导性指标**，应优先报告 AUC-PR 或 F1-score。
- 本论文将 ChestXRay2017 的 Normal/Pneumonia 视为二分类，类别较平衡，但仍需在实验中报告 **全套指标**（Accuracy + AUC + Precision + Recall + F1），以符合学术规范。

**论文位置** 第 2–4 节（各指标定义及适用场景）。

---

## 八、文献引用格式示例（ГОСТ / APA 双格式）

本课程论文使用 **ГОСТ Р 7.0.5–2008**（俄罗斯国家标准）或 **APA 7th**，以下为两种格式示例：

**ГОСТ格式示例：**
> Раджпуркар П., Ирвин Дж., Чжу К. и др. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning // arXiv preprint. 2017. № 1711.05225.

**APA 7th 格式示例：**
> Rajpurkar, P., Irvin, J., Zhu, K., et al. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv preprint arXiv:1711.05225*.

---

## 九、本论文可引用的核心文献清单（优先阅读顺序）

| 序号 | 文献 | 用途 | 必读程度 |
|---|---|---|---|
| 1 | Kermany et al., 2018 (Cell) | 数据集 + 基线方法 | ★★★★★ |
| 2 | Rajpurkar et al., 2017 (arXiv) | 里程碑方法 + 性能对标 | ★★★★★ |
| 3 | Selvaraju et al., 2017 (ICCV) | Grad-CAM 方法原理 | ★★★★★ |
| 4 | He et al., 2016 (CVPR) | ResNet 架构原理 | ★★★★☆ |
| 5 | Stephen et al., 2019 (JHE) | 架构对比实验参考 | ★★★★☆ |
| 6 | Lin et al., 2017 (ICCV) | Focal Loss 原理 | ★★★☆☆ |
| 7 | Woo et al., 2018 (ECCV) | CBAM 原理 | ★★★☆☆ |
| 8 | Wang et al., 2021 (IEEE Access) | 肺炎 X 光综述 | ★★★★☆ |
| 9 | Litjens et al., 2017 (MedIA) | 医学影像 AI 综述 | ★★★☆☆ |

---

## 十、2025–2026 前沿工作（按研究方向分类）

> 以下文献为 DeepSeek 推荐的 2025–2026 年最新研究，用于证明该领域创新空间仍然广阔，
> 并为本课程论文的「创新路径」提供参考。按技术路线分类罗列。

---

### 【方向 A】轻量化 CNN 与边缘部署

#### [L15] Saranyaraj et al., 2026 — PneuNet（轻量化 CNN，直接使用 ChestXRay2017）

**完整出处** Saranyaraj, D., Shrinaath, V., Nayak, A., & Vishal, R. (2026). *PneuNet: a lightweight convolutional neural network with multiscale feature fusion for automated pneumonia detection from chest X-rays*. Frontiers in Medicine (Lausanne), 12, 1713587. https://doi.org/10.3389/fmed.2025.1713587

**数据集** 直接使用 **ChestXRay2017**（Kermany et al. 2018 发布），5,856 张，分层 80/10/10 划分，测试集 624 张。

**关键方法**
- 从零构建轻量化 CNN（不依赖预训练主干），参数量仅 ~1.84M
- 核心模块：深度可分离卷积 + Squeeze-and-Excitation (SE) 块 + Atrous 空间金字塔池化 (ASPP) + 可学习池化层
- 训练：Adam，batch=32，50 epoch，Early stopping (patience=7)

**主要结果（测试集）**
| 指标 | PneuNet | ResNet-50+Attention | DenseNet-121 |
|------|---------|---------------------|---------------|
| Accuracy | 91% | 93% | 90% |
| AUC-ROC | 0.94 | 0.97 | 0.91 |
| 参数量 (M) | 1.8 | 25.6 | 8.0 |
| 推理速度 (ms, RPi4) | 20 | 48 | 62 |

**创新点** 以 1/14 参数量达到接近 SOTA 的性能，可在树莓派 4 上实时推理（<3s 端到端延迟），面向基层诊所低算力场景。

**论文位置** 全文（Frontiers Open Access）

---

#### [L16] Chodagam & Hiremath, 2026 — 显著性特征选择 + 机器学习（IJAI）

**完整出处** Chodagam, Y., & Hiremath, M. (2026). *Pneumonia classification from chest X-rays using significant feature selection and machine learning*. IAES International Journal of Artificial Intelligence (IJAI), 15(1), 592–603. https://doi.org/10.11591/ijai.v15.i1.pp592-603

**关键方法**
- 提取三类放射组学特征：一阶统计特征、纹理特征、小波变换特征
- 特征选择流程：SHAP（可解释性评分）→ RFE（递归特征消除）→ Stability Selection（稳定性筛选）
- 最终分类器使用经典机器学习算法（非深度学习方法），模型极小

**主要结果**
| 数据集 | 准确率 |
|--------|--------|
| 主数据集 | 98% |
| 验证数据集 1 | 97% ± 1% |
| 验证数据集 2 | 96% ± 2% |
| 验证数据集 3 | 94% ± 2% |

**创新点** 不依赖深度学习，用 SHAP+RFE+Stability Selection 选出极少特征即可达到 98% 准确率，模型完全可解释，适合资源受限环境。

**论文位置** 全文（Open Access）

---

### 【方向 B】Vision Transformer 与图 Transformer

#### [L17] Aldabayan, 2026 — ViT 肺炎与气胸二合一检测（PLOS ONE）

**完整出处** Aldabayan, Y. S. (2026). *Pneumonia and pneumothorax detection: A multi-factor evaluation of chest X-rays*. PLOS ONE, 21(1), e0341060. https://doi.org/10.1371/journal.pone.0341060

**数据集** 使用 **NIH ChestX-ray14**（即社区通称的 ChestXRay2017 数据集），仅保留 PA 位片，肺炎 1,431 张（极度不平衡）。

**关键方法**
- 预训练 ViT-Base-Patch16-224，替换分类头为逐类 Sigmoid 输出（支持多标签）
- 处理类别不平衡：WeightedRandomSampler + Focal Loss
- 阈值优化：肺炎 0.05，气胸 0.20（通过验证集确定，非常规 0.5）
- 评估指标明确拒绝 Accuracy，改用 AUC-ROC、AUC-PR、灵敏度、特异度

**主要结果（阈值优化后）**
| 指标 | 肺炎 | 气胸 |
|------|------|------|
| 灵敏度 | ~100% | 97.1% |
| AUC-ROC | 0.5485 | 0.5084 |
| AUC-PR | 0.0247 | 0.1049 |

> 注：AUC 较低的主要原因是 NIH 标签为 NLP 自动生成（弱标签），并非模型本身问题。

**创新点** 首次系统评估 ViT 在胸片肺炎/气胸检测中的多因素表现，建立了医学影像中「拒绝 Accuracy、使用临床指标」的评估规范。

**论文位置** 全文（Open Access）

---

#### [L18] Shan & Yang, 2025 — 多尺度嵌套图 Transformer（MNGT）

**完整出处** Shan, X., & Yang, J. (2025). *Multi-scale nested graph transformer with graph operations: An efficient solution for high-resolution CXR classification*. Medical Physics, 52(3). https://doi.org/10.1002/mp.70003

**关键方法**
- 将高分辨率 CXR 图像构建为图结构，每个节点对应一个图像块
- 多尺度嵌套图 Transformer：在多个尺度上构建图，捕获局部细节与全局上下文
- 图操作（Graph Operations）增强节点间的信息传递，适合小病灶检测

**主要结果** 在高分辨率 CXR 分类任务中达到 SOTA，尤其在小病灶（早期肺炎）检测上优于纯 CNN 方法。（具体数值原文待补）

**创新点** 首次将多尺度嵌套图 Transformer 用于胸片分类，解决了 CNN 在极高分辨率下感受野不足的问题。

**论文位置** Medical Physics（需访问权限， preprint 见 arXiv）

---

### 【方向 C】混合架构与多分支特征

#### [L19] Zahin et al., 2026 — C-RNet（CNN-RNN 混合，Scientific Reports）

**完整出处** Zahin, I. A., Ahsan, M. F., Orni, R. A., Hossain, A., Tabassum, M., Zishan, M. A. O., & Noor, J. (2026). *Enhancing lung diseases recognition through CNN-RNN methodologies*. Scientific Reports, 16, 25422. https://doi.org/10.1038/s41598-026-45842-1

**数据集** COVID-19 Radiography Database（Kaggle，4 类别：Normal / COVID-19 / Pneumonia / Tuberculosis）

**关键方法**
- C-RNet = CNN（空间特征提取）+ RNN（中间层输出的依赖关系建模）
- 中间层特征融合：将 CNN 中间层输出与 RNN 状态显式拼接，再送入分类层
- 可解释性：Grad-CAM 可视化病灶区域

**主要结果**
| 指标 | C-RNet | 对比基线（CNN / RNN 单独） |
|------|---------|--------------------------|
| Accuracy | 93.73% | 显著低于 C-RNet |
| F1-score | 94.6% | — |
| 参数量 | 1.90M | — |
| 模型大小 | 7.25 MB | — |

**创新点** 用 RNN 建模 CNN 中间层的「连续性/依赖性」，突破传统 CNN 单尺度特征提取的局限，参数量极小（1.9M）。

**论文位置** 全文（Nature Open Access）

---

#### [L20] Dahan et al., 2026 — Zone-Aware 肺分区 + 多分支（Springer）

**完整出处** Dahan, F., Shah, J. H., Afzal, M., Aloqaily, M., Almohamedh, R., & Alfakih, T. M. (2026). *Zone-Aware Pneumonia Classification Using Automated Lung Region Detection and Multi-Branch Feature Learning*. International Journal of Computational Intelligence Systems, 19, 134. https://doi.org/10.1007/s44196-026-01208-z

**数据集** Kaggle Chest X-Ray Images (Pneumonia)（ChestXRay2017 的整理子集），5,863 张，Train/Test/Val 划分。

**关键方法**
- YOLOv8 自动检测 5 个肺区：RUZ、RMZ、RLZ、LUZ、LLZ
- 每个肺区独立用 modified ResNet-18 提取特征（前 3 块冻结，第 4 块微调）
- 双阶段 Chain-of-Thought (CoT) 推理：
  - Intra-zone CoT：捕获区内细微模式（磨玻璃影、实变）
  - Inter-zone CoT：注意力加权融合多区特征，检测跨区病变（如双侧渗出）
- 可解释性：逐区 Grad-CAM 热力图

**主要结果**
| 指标 | Zone-Aware 模型 | ResNet-18 | ViT | Hybrid CNN-ViT |
|------|-----------------|-----------|-----|----------------|
| Accuracy | **94.4%** | 86.7% | 84.7% | 94.3% |
| AUC | **0.971** | 0.874 | — | — |
| F1-score | **94.3%** | — | — | 93% |
|  minority recall | **87.7%** | 80.1% | — | — |

**创新点** 首次将胸片按临床肺区进行分区建模，用双阶段 CoT 推理融合跨区信息，Grad-CAM 对齐专家标注达 85.2%。

**论文位置** 全文（Springer Open Access）

---

#### [L21] Li & Li, 2025 — D-BAFN（双分支注意力融合网络）

**完整出处** Li, T., & Li, B. (2025). *Dual-branch attention fusion network for pneumonia detection*. Biomedical Physics & Engineering Express, 11(5). https://doi.org/10.1088/2057-1976/adebf5

**数据集** 儿科肺炎数据集（二分类）+ 多源数据集（三分类：Normal / Pneumonia / COVID-19）

**关键方法**
- 双分支架构：ResNet-18（局部病灶特征）+ Mamba Vision（全局上下文状态空间模型）
- 自注意力融合模块：动态加权两分支特征，而非简单拼接
- 支持二分类（肺炎 vs. 正常）和三分类（Normal / Pneumonia / COVID-19）

**主要结果**
| 任务 | Accuracy | AUC | F1-score |
|------|-----------|-----|-----------|
| 二分类（儿科数据） | 97.78% ± 0.12 | — | — |
| 三分类（多源数据） | 97.20% ± 0.15 | **0.997 ± 0.001** | 0.972 ± 0.006 |

**创新点** 将 CNN 局部特征与状态空间模型（Mamba）全局建模能力融合，自注意力动态融合，三分类 AUC 达 0.997。

**论文位置** 全文（IOP Open Access）

---

### 【方向 D】YOLO 目标检测应用于肺炎定位

#### [L22] Xie et al., 2025 — Fast-YOLO（Frontiers in Neurorobotics）

**完整出处** Xie, Y., Zhu, B., Jiang, Y., Zhao, B., & Yu, H. (2025). *Diagnosis of pneumonia from chest X-ray images using YOLO deep learning*. Frontiers in Neurorobotics, 19, 1576438. https://doi.org/10.3389/fnbot.2025.1576438

**数据集** 重新标注的 MIMIC-CXR 数据集，5 类别（细菌性肺炎 / 病毒性肺炎 / 患病 / 健康 / 肺结核），共 4,194 张。

**关键方法**
- 改进 YOLOv11：集成 C3k2（多尺度特征提取）+ DCNv2（可变形卷积，扩大感受野）+ DynamicConv（输入自适应卷积核）
- 边界框标注（病灶定位），而非仅图像级分类
- 数据增强：Mixup + Mosaic + Copy-Paste + CutMix + Random Erasing

**主要结果（肺炎检测任务）**
| 指标 | Fast-YOLO | YOLOv11 (原始) |
|------|-------------|-------------------|
| Precision | **95.2%** | 94.4% |
| Recall | 94.9% | **95.8%** |
| mAP@0.5 | **97.8%** | 97.6% |
| FPS | **120** | 55 |

**创新点** 将 YOLO 目标检测框架引入肺炎 X 光诊断，可输出病灶边界框（而非黑盒分类），实时性能达 120 FPS，适合临床实时辅助诊断。

**论文位置** 全文（Frontiers Open Access）

---

### 【方向 E】可解释性与可信 AI

#### [L23] Chiang, 2026 — CXR-NeXus（原型记忆 + 反事实一致性，Cureus）

**完整出处** Chiang, L.-F. (2026). *An Interpretable Chest X-ray Classification Framework Using Prototype Memory and Counterfactual Consistency*. Cureus, 18(2), e103134. https://doi.org/10.7759/cureus.103134

**数据集** 公开 4 类 CXR 数据集（Normal / Pneumonia / COVID-19 / Tuberculosis），7,135 张，患者级别无重叠划分。

**关键方法**
- Prototype Memory 模块：每个类别维护多个可学习原型，强迫样本嵌入与同类原型对齐
- 反事实一致性（Counterfactual Consistency）：抑制 Grad-CAM 识别的病灶区域后，疾病置信度应下降
- 证据对齐正则化：惩罚肺外区域的注意力，强迫模型关注解剖合理区域
- 损失函数：分类损失 + 间隔分离 + 原型一致性 + 反事实约束 + 注意力正则

**主要结果（鲁棒性测试）**
| 扰动类型 | CXR-NeXus Accuracy | 最佳基线 Accuracy |
|----------|-------------------|-------------------|
| 亮度/对比度偏移 | ~0.74 | ≤ 0.67 |
| 高斯模糊/噪声 | 最优 | 显著下降 |
| 裁剪/缩放 | 最优 | 显著下降 |

> CXR-NeXus 在干净数据上准确率与基线持平，但在强扰动下显著优于所有基线。

**创新点** 将原型记忆、反事实一致性、注意力约束统一纳入端到端训练目标（无需像素级标注），模型决策行为具备因果可解释性。

**论文位置** 全文（Open Access）

---

#### [L24] Dirichlet 变分信息瓶颈，2026（可信集成学习，待补充全文）

**完整出处** 待补（ScienceDirect，Journal of King Saud University – Computer and Information Sciences 或类似期刊，2026）

**关键方法（基于摘要）**
- 用 Dirichlet 分布对 Chest X-ray 分类结果建模，引入「不确定」(Indeterminate) 类别
- 动态加权多个模型的预测，降低高不确定性样本的错误决策风险
- 变分信息瓶颈框架，同时优化特征压缩与分类性能

**创新点** 让 AI 模型在置信度不足时输出「不确定」而非强行预测，更接近临床医生的决策行为（不确定时建议进一步检查）。

**获取途径** https://doi.org/10.1016/j.jksuci.2025.xxxxx（待补充完整 DOI）

---

### 【方向 F】基准数据集与挑战赛

#### [L25] Dong et al., 2026 — CXR-LT 2026 挑战赛综述（arXiv）

**完整出处** Dong, H., Lin, Y., Zhou, P., et al. (2026). *CXR-LT 2026 Challenge: Multi-Center Long-Tailed and Zero Shot Chest X-ray Classification*. arXiv preprint arXiv:2604.15555. https://arxiv.org/abs/2604.15555

**数据集** 多中心数据集，超过 145,000 张胸片，来源：PadChest + NIH Chest X-ray，开发集和测试集由放射科医生标注（非 NLP 弱标签）。

**关键内容**
- 任务 1：30 个已知病理类别的鲁棒多标签分类（长尾分布）
- 任务 2：6 个未见罕见病的零样本分类（开放世界泛化）
- 评估维度：头尾类别性能差距、模型校准、跨中心泛化
- 结论：视觉-语言基础模型在分布内和零样本任务上均表现最佳；跨中心泛化仍是大挑战

**创新点** 建立了首个同时评估「长尾多标签分类」和「零样本泛化」的胸片 AI 基准，标注质量显著高于以往挑战赛（放射科医生标注 vs. NLP 标签）。

**论文位置** arXiv:2604.15555（开放访问）

---

## 十一、2025–2026 前沿工作对本文的启示（创新路径参考）

| 编号 | 创新路径 | 对应前沿文献 | 与本课程论文的结合点 |
|------|----------|--------------|---------------------|
| A | 轻量化设计（面向边缘部署） | L15 (PneuNet), L16 (IJAI) | 阶段三可选：将 ResNet-18 替换为轻量化架构，测试边缘推理性能 |
| B | ViT / 图 Transformer 对比 | L17 (ViT), L18 (MNGT) | 阶段三：ViT 作为对比架构，验证 CNN 在小数据集上的优势 |
| C | 肺分区建模 | L20 (Zone-Aware) | **可直接扩展**：在 Grad-CAM 可视化中按肺区分析注意力分布 |
| D | 混合架构（CNN+RNN / 双分支） | L19 (C-RNet), L21 (D-BAFN) | 阶段二延伸：用双分支架构融合局部+全局特征 |
| E | 可信 AI（不确定度估计） | L23 (CXR-NeXus), L24 (Dirichlet) | **推荐路径**：在阶段二中加入「不确定样本识别」，输出可信度评分 |
| F | 零样本/跨数据集泛化 | L25 (CXR-LT 2026) | 可作为硕士论文的延伸方向：在一个数据集训练、另一个数据集测试 |

> **优先推荐**：路径 C（肺分区 Grad-CAM 分析）和路径 E（可信度估计），
> 这两者与本课程论文的现有框架（ResNet-18 + Grad-CAM）衔接最自然，且创新点明确。

---

*本节新增于 2026-05-04，基于 DeepSeek 推荐的 2025–2026 年文献补充。*

---

## 十二、更新后的优先阅读清单

| 优先级 | 文献 | 用途 | 必读程度 |
|--------|------|------|----------|
| ★★★★★ | Kermany et al., 2018 (Cell) | 数据集 + 基线 | 已读 |
| ★★★★★ | Rajpurkar et al., 2017 (arXiv) | CheXNet 对标 | 已读 |
| ★★★★★ | Selvaraju et al., 2017 (ICCV) | Grad-CAM 原理 | 已读 |
| ★★★★★ | **PneuNet, 2026 (Front. Med.)** | 最新轻量化 SOTA，直接用 ChestXRay2017 | **新增必读** |
| ★★★★☆ | **CXR-NeXus, 2026 (Cureus)** | 最新可解释性框架 | **新增推荐** |
| ★★★★☆ | **Zone-Aware, 2026 (Springer)** | 肺分区建模，可直接启发阶段二 | **新增推荐** |
| ★★★★☆ | Stephen et al., 2019 (JHE) | 架构对比实验 | 已读 |
| ★★★☆☆ | **D-BAFN, 2025 (BPEE)** | 双分支融合，AUC 0.997 | 可选 |
| ★★★☆☆ | **Fast-YOLO, 2025 (Front. Neurorob.)** | 目标检测定位思路 | 可选 |
| ★★★☆☆ | Wang et al., 2021 (IEEE Access) | 肺炎 X 光综述 | 已读 |

---

*文件更新时间：2026-05-04*
*新增：第十至十二节（2025–2026 前沿工作 及 更新后的优先阅读清单）*
*对应课程论文：Применение глубокого обучения для классификации пневмонии на рентгеновских снимках*
