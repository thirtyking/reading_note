---
tags: []
parent: '[diffusion survey] Diffusion Models: A Comprehensive Survey of Methods and Applications'
collections:
    - '0. 综述'
$version: 2252
$libraryID: 1
$itemKey: 72GR2YXF

---
\[2022-09]\[diffusion survey] Diffusion Models: A Comprehensive Survey of Methods and Applications

# \[diffusion survey] Diffusion Models: A Comprehensive Survey of Methods and Applications

## 概述

你好！很高兴能以这种方式带你解读这篇综述。

**《Diffusion Models: A Comprehensive Survey of Methods and Applications》** 这篇论文是非常经典的入门读物。作为一名“老兵”，我要告诉你的是：不要试图第一遍就把里面引用的几百篇论文都看懂。综述的作用是**地图**，不是**说明书**。

我们现在的目标是带你飞到高空，俯瞰整个“扩散模型”的森林，把路标插好。

***

### 第一阶段：破题与宏观定位 (The "Why" & Context)

#### 1. 领域背景重构：为什么要搞扩散模型？

在 Diffusion 火起来之前（大约2020年以前），生成模型（Generative Models）领域就像是“三国演义”，主要由两大势力统治，但它们都有致命的**痛点**：

*   **GANs (生成对抗网络)**：它是当时的霸主。

    *   *痛点*：**训练极不稳定**。就像两个水平不对等的拳击手（生成器和判别器）打架，很难平衡；而且容易出现**模式坍塌 (Mode Collapse)**——模型变得偷懒，只能生成几张一样的图，失去了多样性。

*   **VAEs (变分自编码器)**：它是理论派的代表。

    *   *痛点*：**画质模糊**。生成的图片总感觉蒙了一层纱，细节丢失严重。

**扩散模型 (Diffusion Models)** 的横空出世，是因为它完美的平衡了这两者：**不仅画质极其精细（超越GAN），而且数学上容易推导、训练极其稳定（不坍塌），生成的样本多样性极高。**

**应用场景**：目前它已经统治了 AIGC (AI Generated Content) 领域。不仅仅是画图（Midjourney, Stable Diffusion），还包括生成视频（Sora）、生成3D资产、甚至用于科学领域的蛋白质结构预测。

#### 2. 发展时间轴 (Timeline)

让我们画一条极简的时间线，记住这几个关键节点：

*   **2015 \[起源]：Deep Unsupervised Learning using Nonequilibrium Thermodynamics (Sohl-Dickstein et al.)**

    *   *地位*：**开山鼻祖，但在当时被忽视了**。作者受非平衡热力学的启发，提出了“先把数据毁掉再复原”的思想，但当时算力和数据都没跟上，效果一般。

*   **2019-2020 \[爆发]：DDPM (Ho et al.) & Score-based Generative Modeling (Song & Ermon)**

    *   *地位*：**Game Changer (规则改变者)**。

    *   DDPM 证明了扩散模型能生成高质量图片。

    *   Song 提出的 Score-based 方法用微分方程统一了视角。这是目前所有主流扩散模型的基石。

*   **2021 \[进化]：Guided Diffusion (OpenAI)**

    *   *地位*：**引入控制**。OpenAI 发现可以通过 Classifier Guidance 让模型按照我们的意愿生成特定类别的图，质量超越了当时的 GAN SOTA。

*   **2022 \[民主化]：Latent Diffusion Models (LDM / Stable Diffusion)**

    *   *地位*：**范式转移 (Paradigm Shift)**。从“在像素空间计算”转移到“在潜空间（Latent Space）计算”。这让模型可以在消费级显卡上运行，直接引爆了全网。

***

### 第二阶段：分类学构建 (Taxonomy & Framework)

如果把扩散模型的研究看作一个文件柜，这篇综述通常会把抽屉分为三层。

#### 1. 核心分类树 (基于建模方式)

虽然现在大家殊途同归，但在早期主要有三个流派，综述里通常会这样分：

1.  **DDPMs (Denoising Diffusion Probabilistic Models)**：

    *   *逻辑*：**离散时间步**。把加噪和去噪过程看作是一步一步走的台阶（比如1000步）。这是最主流的解释方式。

2.  **NCSNs (Noise Conditioned Score Networks)**：

    *   *逻辑*：**基于分数的生成模型**。它不直接预测图像，而是预测数据分布的**梯度 (Score Function)**。想象你在爬山，模型告诉你往哪里走能到达山顶（真实数据分布）。

3.  **SDEs (Stochastic Differential Equations)**：

    *   *逻辑*：**连续时间步**。宋扬（Yang Song）大神把上面两种方法统一成了一个微分方程。把离散的台阶变成了光滑的滑梯。

**资深解读**：作为初学者，**重点关注 DDPM** 的逻辑即可，SDE 是高阶玩家用来做理论分析的工具。

#### 2. 通用数学形式 (极简版)

别被论文里的积分吓跑，扩散模型的核心只干两件事：

1.  **前向过程 (Forward Process, 扩散)**：

    *   不断往图片上撒“高斯噪声”。

    *   公式： $x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon$

    *   *人话*：**毁图不倦**。最后图片变成纯噪声。

2.  **反向过程 (Reverse Process, 采样)**：

    *   训练一个神经网络（通常是 U-Net），让它看一眼带噪的图  $x_t$ ，并问它：“嘿，你觉得刚刚加进去的噪声长什么样？”

    *   **通用优化目标 (Loss Function)**： $L = || \epsilon - \epsilon_\theta(x_t, t) ||^2$

    *   *人话*：**预测噪声**。只要我能预测出每一步加了什么噪声，我减去这个噪声，就能还原出原图。

***

### 第三阶段：经典方法剖析与对比 (Deep Dive & Comparison)

#### 1. 代表性算法精讲：Latent Diffusion Models (LDM)

这是必须要深入理解的，因为它是 Stable Diffusion 的前身。

*   **核心痛点**：之前的 DDPM 直接在像素空间（Pixel Space）操作。生成一张 1024x1024 的图，计算量大得惊人，慢得要死。

*   **核心 Trick**：**感知压缩 (Perceptual Compression)**。

    *   先用一个 VAE (变分自编码器) 把大图片压缩成一个小小的“潜变量 (Latent code)”。

    *   然后在那个小小的潜空间里跑扩散过程。

    *   最后再用 VAE 解码回来。

    *   *比喻*：相当于不直接修缮一栋大楼，而是先画一张蓝图，在蓝图上修改好，再按蓝图施工。效率提升了几十倍。

#### 2. 横向对比：Trade-off 分析

| 特性        | Diffusion Models | GANs (生成对抗网络) | VAEs (变分自编码器) |
| :-------- | :--------------- | :------------ | :------------ |
| **图像质量**  | **极高**(SOTA)     | 高             | 一般 (偏模糊)      |
| **多样性**   | **高**(覆盖全)       | 低 (模式坍塌)      | 高             |
| **训练稳定性** | **极稳**(单纯的回归任务)  | 极差 (对抗博弈)     | 稳             |
| **推理速度**  | **慢**(致命伤)       | **快**(一次前向)   | **快**         |


**Trade-off (核心权衡)**：Diffusion 是用**推理时间**换取了**生成质量**和**多样性**。因为它生成一张图需要反复迭代（比如去噪50次），所以特别慢。这也是现在的研究重点。

***

### 第四阶段：数据集与评估体系 (Benchmarks & Metrics)

#### 1. 主流数据集

*   **CIFAR-10**：学术界的“果蝇”。小图（32x32），用来快速验证算法原型的。

*   **ImageNet**：标准考场。用来证明你的模型在大规模类别下的生成能力。

*   **LAION-5B**：这是工业界的核武器。50亿对图文数据，Stable Diffusion 就是吃这个长大的。

#### 2. 评估指标的陷阱

*   **FID (Fréchet Inception Distance)**：

    *   *定义*：计算生成图片分布和真实图片分布的距离。数值越小越好。

    *   *资深吐槽*：**FID 不等于美学质量**。有时候 FID 很低（分布很接近），但人眼看着觉得很丑。而且 FID 很容易被“刷榜”，有些模型通过过拟合（背答案）来降低 FID。

*   **CLIP Score**：

    *   *定义*：如果你做文生图，这个指标衡量图片和提示词 (Prompt) 的匹配程度。

    *   *注意*：有时候为了追求匹配度，模型会牺牲图像的真实感。

***

### 第五阶段：挑战与未来方向 (Open Problems & Trends)

学生最关心这个，因为这里藏着论文的 Idea。

#### 1. 当前瓶颈 (Pain Points)

*   **推理速度慢**：生成一张图要几秒甚至更久，没法像 GAN 那样实时生成（虽然最近有 Distillation 和 Consistency Models 在解决，但仍是热点）。

*   **可控性 (Controllability)**：虽然有了 ControlNet，但像“我要图里的人左手拿杯子右手拿笔，杯子是红色的笔是蓝色的”这种精细的空间逻辑控制，依然很难做到完美（经常出现“幻觉”或属性错位）。

#### 2. 未来趋势 (Trends)

*   **Video Diffusion (视频生成)**：图片已经卷不动了，现在的热点是视频。难点在于**时序一致性**（Temporal Consistency），即如何保证视频里的人转头时脸不崩。

*   **Multimodal (多模态融合)**：不仅仅是 Text-to-Image，而是 Audio/Image/Video/3D 任意模态互转。

*   **Efficient Architecture (架构革新)**：现在的扩散模型主干多是 U-Net 或 DiT (Diffusion Transformer)。如何设计更高效的架构来替代 Attention 机制（比如 Mamba/SSM 在扩散里的应用）是深水区。

**灌水区预警**：仅仅拿一个新的小数据集跑一遍 Latent Diffusion，这种论文现在很难发顶会了。 **深水区推荐**：从数学本质上加速采样（如 ODE 求解器优化），或者解决扩散模型对训练数据的记忆与隐私泄露问题。

***

### 第六阶段：总结与行动建议 (Takeaways)

#### 1. 一句话总结

扩散模型就是**学习如何优雅地去除噪声**的过程；它牺牲了推理速度，换来了极致的生成质量和训练稳定性。

#### 2. 精读清单 (按顺序读)

综述里引用的几百篇，你只需精读这 4 篇构建骨架：

1.  **\[必读 - 基础] DDPM**: *Denoising Diffusion Probabilistic Models (NeurIPS 2020)* - 理解它是怎么运作的。

2.  **\[必读 - 加速] DDIM**: *Denoising Diffusion Implicit Models (ICLR 2021)* - 理解如何加速采样（从1000步变50步）。

3.  **\[必读 - 工业界标准] LDM**: *High-Resolution Image Synthesis with Latent Diffusion Models (CVPR 2022)* - 理解 Stable Diffusion 的原理。

4.  **\[进阶 - 架构] DiT**: *Scalable Diffusion Models with Transformers (ICCV 2023)* - Sora 的核心架构，理解 Transformer 如何替换 U-Net。

读完这四篇，再回来看这篇综述，你会发现整片森林都在你脚下。加油！

## 思考一

Guided Diffusion (OpenAI) 需要看哪篇论文？

***

这是一个非常精准的问题！"Guided Diffusion" 这个名字很容易引起混淆，因为它既是一个GitHub仓库的名字，又代表了一种特定的技术。

OpenAI 所指的 **Guided Diffusion** 对应的是下面这篇论文：

**论文标题**：**Diffusion Models Beat GANs on Image Synthesis** **作者**：Prafulla Dhariwal, Alex Nichol (OpenAI) **发表时间**：NeurIPS 2021 **代码仓库**：`openai/guided-diffusion` (这就是为什么大家习惯叫它 Guided Diffusion)

***

### 为什么这篇论文很重要？（必读理由）

这篇论文是扩散模型真正**击败 GAN (生成对抗网络)** 的“决战时刻”。你需要重点关注以下三个知识点，它们直接解答了“如何让扩散模型听话”的问题：

#### 1. 核心贡献：Classifier Guidance (分类器引导)

这是这篇论文的灵魂。

*   **问题**：普通的扩散模型生成的图片虽然多样，但很难精确控制它生成什么类别（比如我想生成一只“柯基”，而不是随便一只“狗”）。

*   **解法**：作者提出，在扩散模型反向去噪的过程中，**外挂一个训练好的分类器 (Classifier)**。

    *   每一步去噪时，分类器都会看一眼当前的图，然后算一个梯度（Gradient），告诉扩散模型：“往‘柯基’那个方向走！别走偏了！”

    *   **公式直觉**： $\nabla_{x_t} \log p(x_t) + s \cdot \nabla_{x_t} \log p(y|x_t)$ 。其中  $s$  是引导尺度（Guidance Scale）， $s$  越大，生成的图越像柯基，但多样性越低（Trade-off）。

#### 2. 架构改进 (U-Net Architecture Search)

这篇论文对 U-Net 结构进行了暴力搜索和魔改，确立了后来大家沿用的“黄金标准”：

*   增加模型宽度（Width）比深度（Depth）更有效。

*   引入 **Multi-Head Attention**（多头注意力机制）。

*   使用了 **BigGAN 的 Residual Blocks**。

*   *这也是为什么现在的扩散模型 U-Net 长得都差不多的原因。*

#### 3. 历史地位的辨析：Classifier Guidance vs Classifier-Free Guidance

这也是新手最容易晕的地方，请务必记在小本本上：

*   **OpenAI 这篇 (2021)** 提出的是 **Classifier Guidance**。

    *   *缺点*：你需要额外训练一个能处理噪声图片的分类器，不仅训练麻烦，而且分类器稍微有一点对抗攻击的性质，会导致图片虽然类别对了，但细节有点怪。

*   **后来更流行的是 (2022)**：**Classifier-Free Guidance (CFG)**。

    *   *论文*：*Classifier-Free Diffusion Guidance* (Ho & Salimans, Google Research)。

    *   *区别*：不需要外挂分类器了。在训练扩散模型时，偶尔把“类别标签”扔掉（置空）。推理时，让模型自己对比“有标签预测”和“无标签预测”的差距，并把这个差距放大。

    *   **现状**：Stable Diffusion, DALL-E 2, Midjourney 用的全是 **Classifier-Free Guidance**。

**总结建议**： 先读 OpenAI 这篇 **《Diffusion Models Beat GANs...》** 理解“引导”的数学原理（梯度是如何干预生成的）。读懂了它，你再去理解后来更通用的 Classifier-Free Guidance 就会非常轻松，因为后者只是前者的一个“去分类器化”的优雅变体。
