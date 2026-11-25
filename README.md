# 智能备忘录 Agent

> 云端大模型负责抽取时间与事件，本地朴素贝叶斯完成分类的“端云协同”备忘录。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)

## 简介

传统备忘录要手动填标题、选时间、挑分类，事情一多就容易记混。这个项目把“记录”这一步尽量简化：

- **一句话录入**：输入“明天下午三点去开项目碰头会”，云端大模型自动拆出时间与事件；
- **本地分类**：手机端运行朴素贝叶斯模型，把事项归到工作 / 生活 / 学习；
- **断网可用**：云端不可用时，端侧模型仍能完成分类，不阻塞记录；
- **可人工修正**：分类错了可以在详情页改，改动会持久化到本地。

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [实验结果](#实验结果)
- [增强实验](#增强实验)
- [端侧部署](#端侧部署)
- [说明](#说明)
- [许可证](#许可证)

## 项目结构

```text
.
├── train_model.py         # 训练脚本，生成 model.json 与混淆矩阵
├── svm_compare.py         # 线性 SVM 基线对比
├── predict.py             # 离线推理（不依赖 sklearn）
├── main.py                # 打印依赖版本，核对环境
├── model.json             # 端侧推理权重
├── confusion_matrix.png   # 测试集混淆矩阵
├── requirements.txt
├── experiments/           # 交叉验证、基线对比、学习曲线
│   ├── memo_classifier.py
│   ├── compare_models.py
│   └── results/           # 已附带一次运行结果
└── harmonyos/             # 配套鸿蒙端源码
```

## 快速开始

环境要求：Python 3.10+。

```bash
pip install -r requirements.txt
```

### 1. 训练与评估

```bash
python train_model.py
```

脚本加载内置的 152 条语料（工作 50、生活 51、学习 51），按 8:2 划分，做字符级 TF-IDF 特征提取并训练朴素贝叶斯。运行后控制台输出分类报告，当前目录生成 `model.json` 与 `confusion_matrix.png`。

### 2. 基线对比

```bash
python svm_compare.py
```

输出朴素贝叶斯与线性 SVM 在相同划分下的分类报告，用于说明算法选型。

### 3. 可运行原型

`predict.py` 直接读取 `model.json`，在本地复现端侧的加权求和与 argmax 逻辑，不需要 sklearn 环境也能跑：

```bash
python predict.py "明天下午三点开项目碰头会"
# 预测分类: 工作

python predict.py "下班去超市买点牛奶"
# 预测分类: 生活

python predict.py "复习机器学习第六章"
# 预测分类: 学习
```

## 实验结果

| 项目 | 结果 |
| :--- | :--- |
| 数据集 | 152 条人工构造语料，8:2 划分，测试集 31 条 |
| 特征 | 字符级 TF-IDF，1-gram / 2-gram |
| 朴素贝叶斯 | Accuracy 0.84，macro-F1 0.84 |
| 分类表现 | 生活类 F1=0.95；工作 / 学习存在少量互判 |
| SVM 基线 | 与 NB 接近（同为 0.84） |
| 部署成本 | model.json 约 116 KB，单条推理约 20 µs |

误差主要来自长句：例如“下周三下午两点跟老王去北京出差，记得带合同”，其中“去、带、记得”等中性词稀释了“出差”的权重，被误判为生活类。这一点在报告的误差分析里有详细讨论。

## 增强实验

`experiments/` 下是补充验证脚本，已附带一次运行结果，重新运行会在 `experiments/` 生成 `artifacts/` 与 `artifacts_compare/`。这些实验不影响根目录的端侧模型格式。

```bash
cd experiments
python memo_classifier.py --ablation   # 5 折交叉验证 + 填充词剥离消融
python compare_models.py               # 基线对比 + 5×10 重复 CV + 学习曲线
```

- `results/comparison.md`：多数类 / 关键词规则 / NB / SVM 的 macro-F1、95% CI、Wilcoxon p 值；
- `results/learning_curve.png`：macro-F1 随训练样本数的变化；
- `results/errors.csv`：折外预测错误样例。

注意：`memo_classifier.py` 导出的 `artifacts/model.json` 是带 idf 的新格式，仅用于离线一致性校验，不能替换根目录的 `model.json`（App 端 `MlEngine.ets` 读取的是旧格式）。

## 端侧部署

`harmonyos/` 是配套的鸿蒙 App，用于验证模型在真实设备上的落地；这一节不属于 ML 课程评分必需步骤，只跑 Python 部分也能完成实验。

1. 把根目录的 `model.json` 复制到 `harmonyos/entry/src/main/resources/rawfile/`，仓库里已放了一份，重新训练后记得覆盖；
2. 打开 `harmonyos/entry/src/main/ets/utils/LlmService.ets`，把 `API_KEY` 换成自己的智谱开放平台 Key（仓库中为占位符）；
3. 用 DevEco Studio 打开 `harmonyos/`，等待 Sync 完成，连接模拟器后点击 Run；
4. 在 App 中新建备忘，输入“明天下午三点去开项目碰头会”，点击“运行 AI Agent 解析”，可以看到云端抽取的时间 / 事件与端侧分类结果；保存后可在首页搜索、修改、删除。

## 说明

- 开发中使用 Kimi、DeepSeek 等大模型辅助编写正则与基础代码框架，所有生成内容均经本人修改并在本地验证；端侧推理、异常降级等核心逻辑由本人完成。
- 第三方库：Scikit-learn、Pandas、Matplotlib、Seaborn，均遵循各自的开源协议。

## 许可证

本项目代码遵循 MIT 许可证。
