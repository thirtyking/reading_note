---
tags: []
parent: '[Agent入门]LLM Powered Autonomous Agents | Lil''Log'
collections:
    - 智能体
$version: 2206
$libraryID: 1
$itemKey: ZCGVRRYI

---
\[2023-06]\[Agent入门]LLM Powered Autonomous Agents

# LLM Powered Autonomous Agents

<https://lilianweng.github.io/posts/2023-06-23-agent/>

这是一篇由 OpenAI 的 Lilian Weng 撰写的非常经典的博文，题目为《LLM-powered Autonomous Agents》（大语言模型驱动的自主代理）。这篇文章被广泛认为是 **AI Agent（智能体）领域的入门必读指南**。

她提出了一个核心公式：**Agent = LLM（大脑） + Planning（规划） + Memory（记忆） + Tool Use（工具使用）**。

以下是该文章的详细内容总结：

### 1. 核心架构概览

文章将 LLM 定义为代理的“大脑”，并辅以三个关键组件来补全其能力：

*   **规划（Planning）：** 子目标分解、反思与改进。
*   **记忆（Memory）：** 短期记忆（上下文）与长期记忆（向量数据库）。
*   **工具使用（Tool Use）：** 调用外部 API 获取信息或执行操作。

***

### 2. 详细组件解析

#### A. 规划 (Planning)

LLM 需要将复杂的任务转化为可执行的步骤。

*   **任务分解 (Task Decomposition)：**

    *   **思维链 (Chain of Thought, CoT)：** 让模型一步步思考。
    *   **思维树 (Tree of Thoughts, ToT)：** 探索多种推理路径。
    *   **LLM+P：** 将规划外包给外部求解器。

*   **自我反思 (Self-Reflection)：**

    *   **ReAct (Reason+Act)：** 在行动前进行推理，行动后观察结果。
    *   **Reflexion：** 记录过去的失败，生成“自我反思”存储在记忆中，以避免重蹈覆辙。
    *   \*\*Chain of Hindsight (CoH)：\*\*通过提供正负样本反馈来改进输出。

#### B. 记忆 (Memory)

为了克服 LLM 上下文窗口的限制，需要构建记忆系统。

*   **记忆类型：**

    *   **感觉记忆：** 原始输入（如文本、图像）。
    *   **短期记忆：** 模型的上下文窗口（Context Window）。
    *   **长期记忆：** 外部向量数据库（Vector Database），用于存储和检索长时间跨度的信息。

*   **检索机制：**

    *   使用 **Embedding（嵌入）** 将文本转化为向量。
    *   使用 **MIPS（最大内积搜索）** 等算法根据相关性、新近度（Recency）和重要性来检索记忆。

#### C. 工具使用 (Tool Use)

LLM 自身只有预训练的知识，容易产生幻觉且知识过时。工具使用赋予了它“手”和“眼”。

*   **MRKL 系统：** 结合专家系统和 LLM。
*   **Toolformer：** 自我监督学习如何调用 API。
*   **HuggingGPT：** 使用 ChatGPT 作为控制器，调度 HuggingFace 上的各种模型来处理多模态任务。
*   **API 调用：** 学习如何生成参数并解析 API 的返回值。

***

### 3. 案例研究 (Case Studies)

文章列举了几个典型的 Agent 实现：

*   **科学发现代理 (ChemCrow)：** 结合化学工具搜索、规划和解释数据。
*   **生成式代理 (Generative Agents)：** 即著名的“斯坦福 25 人小镇实验”。通过观察、规划和反思机制，模拟了类似人类的社会行为（如传播消息、组织派对）。
*   **概念验证项目：** 如 **AutoGPT**、**GPT-Engineer** 和 **BabyAGI**。这些项目展示了如何通过递归循环（LLM 自己给自己下达指令）来完成复杂目标。

***

### 4. 挑战与局限 (Challenges)

尽管前景广阔，但目前的 Agent 系统仍面临主要问题：

*   **有限的上下文长度：** 虽然向量数据库能辅助，但无法完全替代无限的上下文窗口，历史信息的丢失会影响推理。
*   **长期规划与任务分解的困难：** 面对非常复杂的任务，LLM 容易在漫长的规划链条中“迷路”或陷入死循环，无法从错误中恢复。
*   **自然语言接口的可靠性：** LLM 有时无法输出格式完美的指令（如 JSON 格式错误），导致与工具对接失败。

### 总结

Lilian Weng 的这篇文章系统性地梳理了基于 LLM 的 Agent 的设计范式。通过将 **LLM 作为通用问题解决器（大脑）**，并外挂 **存储（记忆）** 和 **接口（工具）**，我们正在接近通用人工智能（AGI）的雏形，但目前在稳定性上仍有很长的路要走。
