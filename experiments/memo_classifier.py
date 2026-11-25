#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""备忘录短句三分类（工作 / 生活 / 学习），训练 + 对比 + 端侧导出。

训练与评估：
    python memo_classifier.py
    python memo_classifier.py --folds 10 --seed 7 --ablation

用导出的权重离线预测（不依赖 sklearn）：
    python memo_classifier.py --predict "明天下午要交周报"

主要设计取舍：
    1. 样本平均 8.8 个字，词级分词在 152 条规模上边界不稳，改用字符 n-gram；
    2. 字符 n-gram 会把空格、全角标点当特征，所以先做 NFKC 归一并只保留字母数字，
       "背 50 个英语单词" 和 "背50个英语单词" 会落到同一个特征空间；
    3. 每类只有 50~51 条，单次 8:2 划分的测试集仅 30 条，macro-F1 抖动可达 10 个点，
       因此用分层 5 折交叉验证报告均值 ± 标准差，混淆矩阵取折外预测；
    4. 端侧只做加权求和 + argmax，导出的权重必须能和离线 pipeline 逐条对齐，
       脚本末尾的 parity 校验就是为此存在：对不上就报错，而不是悄悄出错。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# 语料：每行一条备忘录。用三行缩进块代替 (句子, 标签) 元组列表，
# 增删一条只需要改一行，也不会因为手写元组漏逗号。
MEMOS: dict[str, str] = {
    "工作": """
明天下午三点开项目碰头会
把季度报表发给王总
准备周一的 PPT 汇报
和客户确认合同细节
修改项目需求文档
参加部门周会
回复重要工作邮件
整理项目进度表
和团队讨论技术方案
提交本周的工作周报
对接设计稿的修改意见
排查线上的紧急 bug
组织面试候选人的会议
整理报销单并提交审批
跟进供应商的交付进度
筹备下个月的发布会
与产品经理对齐排期
写一份竞品分析报告
处理客户的投诉工单
培训新入职的同事
开跨部门的协调会
优化核心业务的代码
测试新版本的更新包
盘点部门的固定资产
审核下属提交的方案
参加行业交流峰会
更新公司官网的动态
跟财务核对预算情况
整理会议纪要并发送
催办停滞的业务流程
制定下个季度的目标
准备年终汇报的材料
和法务确认合同条款
整理客户联系名单
去机房检查服务器状态
配合审计部门提供数据
规划团队的团建活动
写代码实现登录模块
修复页面布局错乱的问题
做一次技术分享演讲
评估外包团队的工作质量
回复老板的微信留言
梳理业务的核心链路
办理项目的结项手续
沟通服务器的采购事宜
处理知识产权的申请
录入新的客户资料
跟运营同步活动规则
审查系统的安全漏洞
安排下一阶段的任务分工
""",
    "生活": """
下班去超市买点牛奶
周末约朋友去吃火锅
交一下这个月的电费
去取个快递
晚上去健身房锻炼
给家里打个电话
预约周末理发
买两件换季衣服
打扫房间卫生
记得去物业拿门禁卡
给汽车加满油
去药店买点感冒药
去菜市场买点排骨炖汤
把冬天的被子拿出来晒
预约牙医洗牙
帮老妈买个新手机
周末去郊区露营
去银行办一张储蓄卡
买一箱矿泉水放冰箱
洗一下积攒的脏衣服
去营业厅换手机套餐
给猫买两袋猫粮
晚上看一部新出的电影
买点水果带回家
把不穿的衣服捐掉
去花鸟市场买两盆绿植
换一下床单被套
去交宽带网费
买点感冒药备用
修一下漏水的水龙头
周末带家人去公园野餐
去干洗店拿大衣
买一副新的隐形眼镜
去吃一顿烤肉自助
去超市采购下周的食材
把换季的鞋子刷干净
在网上下单买点零食
去社区医院打疫苗
帮室友带份午饭
清理一下手机内存
预约空调上门清洗服务
去洗个车
买一个收纳箱整理杂物
去夜市逛逛吃点小吃
给手表换一块电池
去快递站寄个包裹
买张新的电话卡
去公园散步跑两圈
把冰箱里的过期食物扔掉
去理发店烫个头发
给父母买点保健品
""",
    "学习": """
复习机器学习第六章
背 50 个英语单词
看一篇关于大模型的论文
完成 Python 课后作业
准备期末考试复习计划
听一场学术讲座
整理课堂笔记
刷两套六级真题
看网课学习鸿蒙开发
写数据结构实验报告
预习明天的算法课
看完教材第三章
刷力扣上的算法题
准备考研数学的复习
听英语听力磨耳朵
做一个课程设计的小项目
查阅相关的参考文献
准备明天的小组汇报
总结今天学到的知识点
完成网课视频的打卡
写一篇读书笔记
复习高等数学公式
做一套计算机网络的卷子
学习大模型微调的方法
搭建深度学习的环境
看一部TED演讲视频
尝试复现一篇经典的论文
向老师请教问题
把错题整理到错题本上
练习英语口语
看一集技术教学视频
梳理操作系统核心概念
参加线下的技术沙龙
写博客记录今天的学习
分析数据集的分布情况
复习概率论的基础知识
读一本专业相关的课外书
准备下个月的软考
看开源代码的底层实现
做一份课程汇报的PPT
背一下政治的考点
做一套英语阅读理解的练习
学习前端框架的基础
搭建一个个人网站
写完实验二的报告
复习微观经济学的概念
在B站看考研数学的网课
跟室友交流学习心得
准备英语四级的作文
做一下算法课的上机题
复习期末考点
""",
}

# 口语填充词。中文短句里"一下/记得"几乎不携带类别信息，
# 却会因为高频而占掉词表，剥掉它们对端侧词表体积也有好处。
FILLERS: tuple[str, ...] = ("记得", "一下", "一点", "帮我", "顺便", "记得要")

NGRAM_RANGE: tuple[int, int] = (1, 3)
PADDING = ("^", "$")
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
OUT_DIR = Path("artifacts")
CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
)


# ---------------------------------------------------------------- 特征部分


def normalize(text: str, drop_fillers: bool = True) -> str:
    """全角转半角、小写、只留字母数字，去掉口语填充词。

    NFKC 会把全角括号、全角数字、"《》" 之类的变体统一，
    否则 "50" 和 "５０" 会被当成两种不同的字符特征。
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    kept = "".join(ch for ch in folded if ch.isalnum())
    if drop_fillers:
        for filler in FILLERS:
            kept = kept.replace(filler, "")
    return kept


def char_ngrams(text: str, lo: int = NGRAM_RANGE[0], hi: int = NGRAM_RANGE[1]) -> list[str]:
    """在加边界符的串上切 1~3 元字符片段。

    加边界符是为了让 "复习" 与 "复习了" 的片段不至于完全相同，
    短文本里首尾两个字的区分度往往比中间更高。
    """
    padded = PADDING[0] + text + PADDING[1]
    grams: list[str] = []
    for size in range(lo, hi + 1):
        grams.extend(padded[i : i + size] for i in range(len(padded) - size + 1))
    return grams


def make_analyzer(drop_fillers: bool):
    """给 TfidfVectorizer 用的 analyzer：先归一化再切片段。"""

    def analyze(text: str) -> list[str]:
        return char_ngrams(normalize(text, drop_fillers))

    return analyze


def load_corpus() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for label, block in MEMOS.items():
        for line in block.splitlines():
            memo = line.strip()
            if memo:
                texts.append(memo)
                labels.append(label)
    return texts, labels


# ---------------------------------------------------------------- 训练部分


def build_pipeline(clf, drop_fillers: bool):
    # 延迟导入：只有真正训练时才需要 sklearn，
    # 这样 normalize / char_ngrams / predict_exported 可以单独 import 做对齐测试。
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=make_analyzer(drop_fillers),
                    sublinear_tf=True,  # 1+log(tf)，抑制长句里的重复字
                    norm="l2",
                    lowercase=False,  # 归一化已在前面的 analyzer 里做过
                ),
            ),
            ("clf", clf),
        ]
    )


def candidate_models(seed: int) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB, MultinomialNB
    from sklearn.svm import LinearSVC

    return {
        "MultinomialNB(alpha=0.2)": MultinomialNB(alpha=0.2),
        "MultinomialNB(alpha=0.5)": MultinomialNB(alpha=0.5),
        "MultinomialNB(alpha=1.0)": MultinomialNB(alpha=1.0),
        "ComplementNB(alpha=0.3)": ComplementNB(alpha=0.3),
        "LinearSVC(C=1.0)": LinearSVC(C=1.0, random_state=seed),
        "LogisticRegression(C=5.0)": LogisticRegression(C=5.0, max_iter=2000, random_state=seed),
    }


def cross_validate_candidates(texts, labels, seed: int, folds: int, drop_fillers: bool):
    """对所有候选模型做同一套分层折，返回 [(名字, acc均值, acc标准差, f1均值, f1标准差)]。"""
    from sklearn.model_selection import StratifiedKFold, cross_validate

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for name, clf in candidate_models(seed).items():
        pipe = build_pipeline(clf, drop_fillers)
        scores = cross_validate(
            pipe, texts, labels, cv=splitter, scoring=["accuracy", "f1_macro"], n_jobs=1
        )
        rows.append(
            (
                name,
                scores["test_accuracy"].mean(),
                scores["test_accuracy"].std(),
                scores["test_f1_macro"].mean(),
                scores["test_f1_macro"].std(),
            )
        )
    return rows


def print_leaderboard(rows) -> None:
    header = f"{'模型':<28}{'accuracy':>12}{'macro-F1':>14}"
    print(header)
    print("-" * len(header))
    for name, acc, acc_std, f1, f1_std in sorted(rows, key=lambda r: r[3], reverse=True):
        print(f"{name:<28}{acc:>8.3f}±{acc_std:<5.3f}{f1:>10.3f}±{f1_std:<5.3f}")


def fit_exportable_model(texts, labels, seed: int, folds: int, drop_fillers: bool):
    """挑一个 MultinomialNB 变体并在全量数据上重训。

    端侧公式是 log_prior + Σ tfidf·log_prob，只有 MultinomialNB 与之逐项对应；
    ComplementNB 的决策函数不带先验，LinearSVC 更没有可用的概率权重。
    所以"对比实验"可以报任意模型，"导出"固定走 MultinomialNB，并在返回时说明。
    """
    rows = [r for r in cross_validate_candidates(texts, labels, seed, folds, drop_fillers) if r[0].startswith("MultinomialNB")]
    best_name, _, _, best_f1, _ = max(rows, key=lambda r: r[3])
    from sklearn.naive_bayes import MultinomialNB

    alpha = float(best_name.split("=")[1].rstrip(")"))
    pipe = build_pipeline(MultinomialNB(alpha=alpha), drop_fillers)
    pipe.fit(texts, labels)
    return pipe, best_name, best_f1


def evaluate_out_of_fold(pipe, texts, labels, seed: int, folds: int):
    """折外预测：每个样本都由没见过它的那一折模型判分，结果才能可靠地画混淆矩阵。"""
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predicted = cross_val_predict(pipe, texts, labels, cv=splitter, n_jobs=1)
    labels_sorted = sorted(set(labels))
    matrix = confusion_matrix(labels, predicted, labels=labels_sorted)
    report = classification_report(labels, predicted, digits=3)
    return matrix, report, labels_sorted


def plot_confusion(matrix, class_names, path: Path) -> bool:
    """画混淆矩阵；没有 matplotlib 或没有中文字体时降级为 CSV，而不是崩掉。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import font_manager, pyplot as plt
    except ImportError:
        print("[warn] 未安装 matplotlib，混淆矩阵改为写 csv")
        return False

    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in CJK_FONT_CANDIDATES if name in available), None)
    if chosen is None:
        print(f"[warn] 系统里没有可用中文字体（查过 {', '.join(CJK_FONT_CANDIDATES)}），标签可能显示为方块")
    else:
        plt.rcParams["font.sans-serif"] = [chosen, *CJK_FONT_CANDIDATES]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    image = ax.imshow(matrix, cmap="YlGnBu")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"折外混淆矩阵（{class_names[0]} / {class_names[1]} / {class_names[2]}）")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            # 深色格用白字，浅色格用黑字，比统一白字清楚
            color = "white" if value > matrix.max() * 0.55 else "#22303f"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- 端侧导出


def export_weights(pipe, drop_fillers: bool, meta: dict) -> dict:
    """把 pipeline 里"推理真正需要的四张表"拆出来，去掉 sklearn 依赖。

    四张表：词表 index、idf、各类 log_prob、类先验 log_prior。
    端侧少做一步 sublinear_tf 或漏了 idf，结果就和离线对不上。
    """
    vectorizer = pipe.named_steps["tfidf"]
    nb = pipe.named_steps["clf"]
    vocabulary = vectorizer.vocabulary_
    # get_feature_names_out 的顺序与 vocabulary_ 的下标一致，idf_ 也按这个顺序排
    features = [name for name, _ in sorted(vocabulary.items(), key=lambda kv: kv[1])]
    return {
        "format": "memo-nb-v1",
        "labels": [str(name) for name in nb.classes_],
        "ngram_range": list(NGRAM_RANGE),
        "normalize": {"nfkc": True, "drop_fillers": list(FILLERS) if drop_fillers else [], "keep": "alnum"},
        "features": features,
        "idf": [round(float(v), 6) for v in vectorizer.idf_],
        "log_prior": [round(float(v), 6) for v in nb.class_log_prior_],
        "log_prob": [[round(float(v), 6) for v in row] for row in nb.feature_log_prob_],
        "trained_with": meta,
    }


def predict_exported(model: dict, text: str) -> tuple[str, list[float]]:
    """只用导出的四张表复现 MultinomialNB + TF-IDF 的推理过程。"""
    index = {feature: i for i, feature in enumerate(model["features"])}
    idf = model["idf"]
    drop_fillers = bool(model["normalize"]["drop_fillers"])

    counts = Counter(char_ngrams(normalize(text, drop_fillers)))
    vector = [0.0] * len(idf)
    for feature, count in counts.items():
        position = index.get(feature)
        if position is not None:
            vector[position] = (1.0 + math.log(count)) * idf[position]

    length = math.sqrt(sum(value * value for value in vector))
    if length > 0:
        vector = [value / length for value in vector]

    scores = []
    for prior, weights in zip(model["log_prior"], model["log_prob"]):
        scores.append(prior + sum(v * w for v, w in zip(vector, weights) if v))
    best = max(range(len(scores)), key=scores.__getitem__)
    return model["labels"][best], scores


def verify_parity(pipe, model: dict, texts, labels, drop_fillers: bool) -> float:
    """逐条对比"pipeline.predict"与"只用 JSON 推理"，不一致就报错退出。"""
    offline = pipe.predict(texts)
    mismatched = []
    for text, expected in zip(texts, offline):
        got, _ = predict_exported(model, text)
        if got != expected:
            mismatched.append((text, expected, got))
    if mismatched:
        print(f"[fail] 端侧公式与离线结果不一致 {len(mismatched)} 条，导出不可信：")
        for text, expected, got in mismatched[:5]:
            print(f"  - {text}: 离线={expected} 端侧={got}")
        sys.exit(1)
    return 1.0


# ---------------------------------------------------------------- 入口


def run_ablation(texts, labels, seed: int, folds: int) -> None:
    """剥填充词到底有没有用，用同一套折做对照，比拍脑袋说"效果好"可靠。"""
    print("\n[ablation] 填充词剥离前后对比（均为 MultinomialNB(0.5) + 字符 1~3 元）")
    for drop in (False, True):
        rows = [r for r in cross_validate_candidates(texts, labels, seed, folds, drop) if r[0] == "MultinomialNB(alpha=0.5)"]
        name, acc, acc_std, f1, f1_std = rows[0]
        flag = "剥离" if drop else "保留"
        print(f"  {flag}: accuracy={acc:.3f}±{acc_std:.3f}  macro-F1={f1:.3f}±{f1_std:.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="备忘录短句三分类训练与端侧导出")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子，影响折划分与模型")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="分层交叉验证折数")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="产物目录")
    parser.add_argument("--predict", type=str, help="用导出权重预测一句话，不重新训练")
    parser.add_argument("--ablation", action="store_true", help="额外跑填充词剥离的对照实验")
    parser.add_argument("--keep-fillers", action="store_true", help="训练时不剥离口语填充词")
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    model_path = out_dir / "model.json"
    drop_fillers = not args.keep_fillers

    if args.predict:
        if not model_path.exists():
            print(f"[fail] 找不到 {model_path}，先跑一次训练", file=sys.stderr)
            return 2
        model = json.loads(model_path.read_text(encoding="utf-8"))
        label, scores = predict_exported(model, args.predict)
        ranked = sorted(zip(model["labels"], scores), key=lambda kv: kv[1], reverse=True)
        print(f"输入：{args.predict}")
        print(f"判定：{label}")
        for name, score in ranked:
            print(f"  {name}: {score:.3f}")
        return 0

    texts, labels = load_corpus()
    counts = Counter(labels)
    print(f"语料 {len(texts)} 条：" + "，".join(f"{k} {v}" for k, v in counts.items()))
    duplicates = [text for text, times in Counter(texts).items() if times > 1]
    if duplicates:
        print(f"[warn] 有 {len(duplicates)} 条重复样本，训练前建议去重：{duplicates[:3]}")

    print(f"\n[1/4] 候选模型 {args.folds} 折交叉验证")
    leaderboard = cross_validate_candidates(texts, labels, args.seed, args.folds, drop_fillers)
    print_leaderboard(leaderboard)

    if args.ablation:
        run_ablation(texts, labels, args.seed, args.folds)

    print("\n[2/4] 选端侧可导出的模型并在全量数据上重训")
    pipe, best_name, best_f1 = fit_exportable_model(texts, labels, args.seed, args.folds, drop_fillers)
    print(f"  导出模型：{best_name}（该配置折外 macro-F1={best_f1:.3f}）")

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix, report, class_names = evaluate_out_of_fold(pipe, texts, labels, args.seed, args.folds)
    print("\n[3/4] 折外评估")
    print(report)

    figure_path = out_dir / "confusion_matrix.png"
    if plot_confusion(matrix, class_names, figure_path):
        print(f"  混淆矩阵：{figure_path}")
    else:
        csv_path = out_dir / "confusion_matrix.csv"
        rows = [",".join(["真实\\预测", *class_names])]
        rows += [",".join([name, *map(str, matrix[i])]) for i, name in enumerate(class_names)]
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"  混淆矩阵：{csv_path}")

    print("\n[4/4] 导出端侧权重并校验一致性")
    model = export_weights(
        pipe,
        drop_fillers,
        meta={
            "model": best_name,
            "samples": len(texts),
            "cv_macro_f1": round(float(best_f1), 4),
            "folds": args.folds,
            "seed": args.seed,
        },
    )
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    size_kb = model_path.stat().st_size / 1024
    print(f"  权重：{model_path}（{len(model['features'])} 个特征，{size_kb:.1f} KB）")
    verify_parity(pipe, model, texts, labels, drop_fillers)
    print(f"  一致性校验通过：{len(texts)} 条样本离线/端侧结果完全相同")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
