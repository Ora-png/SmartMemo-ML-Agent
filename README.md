# 智能备忘录 Agent：基于端云协同的文本分类系统

本项目是《机器学习》课程的期末大作业。项目针对传统备忘录手动分类繁琐的痛点，设计并实现了一款“端云协同”的智能备忘录 Agent。

系统架构分为两部分：云端调用大语言模型（智谱 GLM）提取自然语言中的时间与事件实体；本地端侧运行**朴素贝叶斯（Naive Bayes）**机器学习模型，实现零网络延迟的智能分类。本仓库包含模型训练、对比实验、离线推理脚本，以及配套的 HarmonyOS 端侧源码。

## 项目目录结构

```text
.
├── train_model.py         # 训练脚本，生成 model.json 和混淆矩阵
├── svm_compare.py         # 基线对比（朴素贝叶斯 vs 线性 SVM）
├── predict.py             # 离线推理脚本（可运行原型，不依赖 sklearn）
├── main.py                # 打印依赖库版本，核对环境
├── model.json             # 训练导出的端侧模型权重
├── confusion_matrix.png   # 测试集混淆矩阵
├── requirements.txt       # Python 依赖
├── .gitignore
├── README.md
├── experiments/           # 补充实验：交叉验证、基线对比、学习曲线
│   ├── memo_classifier.py
│   ├── compare_models.py
│   └── results/           # 已附带一次运行结果
└── harmonyos/             # 配套鸿蒙端源码（可选运行）
```

## 环境安装

请确保已安装 Python 3.10+，然后在终端运行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

依赖版本为：pandas 3.1.1、scikit-learn 1.9.1、matplotlib 3.11.2、seaborn 0.13.2。

## 运行步骤

### 1. 模型训练与评估

```bash
python train_model.py
```

脚本会加载内置的 152 条语料（工作 50 条、生活 51 条、学习 51 条），按 8:2 划分训练集与测试集，做字符级 TF-IDF 特征提取并训练朴素贝叶斯。运行后控制台会输出分类报告，并在当前目录生成 `model.json` 与 `confusion_matrix.png`。

### 2. 基线对比实验

```bash
python svm_compare.py
```

输出朴素贝叶斯与线性 SVM 在相同划分下的分类报告，用于说明算法选型依据。

### 3. 可运行原型：命令行推理

`predict.py` 直接读取 `model.json`，在本地复现端侧的加权求和与 argmax 逻辑，不需要 sklearn 环境也能跑：

```bash
python predict.py "明天下午三点开项目碰头会"
# 预期输出：预测分类: 工作

python predict.py "下班去超市买点牛奶"
# 预期输出：预测分类: 生活

python predict.py "复习机器学习第六章"
# 预期输出：预测分类: 学习
```

### 4. 鸿蒙端运行（可选）

1. 将根目录的 `model.json` 复制到 `harmonyos/entry/src/main/resources/rawfile/`（仓库中已放好一份，重新训练后需覆盖）。
2. 打开 `harmonyos/entry/src/main/ets/utils/LlmService.ets`，把 `API_KEY` 换成自己的智谱开放平台 Key（仓库中为占位符）。
3. 用 DevEco Studio 打开 `harmonyos/` 目录，等待 Sync 完成，连接模拟器后点击 Run。
4. 在 App 中新建备忘，输入“明天下午三点去开项目碰头会”，点击“运行 AI Agent 解析”，可看到云端抽取的时间/事件与端侧分类结果；保存后可在首页搜索、修改、删除。

## 增强实验（可选）

`experiments/` 下是补充验证脚本；仓库中已附带一次运行结果（`experiments/results/`），重新运行会在 `experiments/` 下生成 `artifacts/` 与 `artifacts_compare/`。这些实验不影响根目录的端侧模型格式：

```bash
cd experiments
python memo_classifier.py --ablation
python compare_models.py
```

- `comparison.md`：多数类 / 关键词规则 / NB / SVM 的 macro-F1、95% CI、Wilcoxon p 值；
- `learning_curve.png`：macro-F1 随训练样本数的变化；
- `errors.csv`：折外预测错误样例，用于误差分析。

注意：`memo_classifier.py` 导出的 `artifacts/model.json` 是带 idf 的新格式，仅用于离线一致性校验，不能替换根目录的 `model.json`（App 端 `MlEngine.ets` 读取的是旧格式）。
## 实验结果与核心数据

* **数据集规模**：152 条人工构造语料（工作 50 条、生活 51 条、学习 51 条），按 8:2 划分，测试集 31 条。
* **特征工程**：字符级（Char-level）TF-IDF 与 1-gram/2-gram，不依赖外部中文分词库，降低端侧实现成本。
* **模型表现**：朴素贝叶斯在测试集上准确率 **0.84**，宏平均 F1 为 **0.84**；其中“生活”类表现最好（F1=0.95），测试集上线性 SVM 的结果与 NB 接近（同为 0.84），因此从模型体积和推理开销考虑选择了 NB。
* **误差分析**：混淆矩阵显示“工作”与“学习”之间存在少量互判；长句“下周三下午两点跟老王去北京出差”因“去、带、记得”等中性词稀释了“出差”这一强特征，被误判为“生活”，暴露出字袋模型缺乏上下文语义的局限。

详细的误差分析、SVM 对比结论与端侧落地细节，见提交的课程报告 PDF。

## 实际落地演示（HarmonyOS）

训练得到的 `model.json` 已部署到 HarmonyOS 端侧，端侧用 ArkTS 实现了与 `predict.py` 一致的推理逻辑（单字 + 双字切分、查词表、累加对数概率、取最大类）。端侧代码位于 `harmonyos/entry/src/main/ets/utils/MlEngine.ets`；云端请求封装在 `LlmService.ets`，网络异常时会提示并降级到本地分类。

## AI 工具使用说明

本项目在开发与报告撰写过程中，使用 Kimi、DeepSeek 等大模型辅助生成部分正则表达式与基础代码框架。所有 AI 生成代码均已由本人重新理解、修改，并在本地跑通验证；核心的端侧推理数学逻辑、异常降级处理等由本人独立完成。

## 开源许可

本项目代码遵循 MIT 许可证。所使用的第三方库（Scikit-learn、Pandas、Matplotlib、Seaborn）均遵循其对应的 BSD/PSF 开源协议。