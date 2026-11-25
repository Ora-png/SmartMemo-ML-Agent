#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比实验：逐级基线 + 重复交叉验证 + 配对检验 + 学习曲线 + 部署成本。

和 memo_classifier.py 放在同一目录，复用其内置语料。

用法：
    python compare_models.py

产出（artifacts_compare/）：
    comparison.csv       每个配置的折得分与汇总
    comparison.md        模型对比表（Markdown）
    learning_curve.png   macro-F1 随训练集规模的变化
    errors.csv           折外预测的全部错误，供误差分析归类

设计要点：
    1. 31 条测试集上单个模型的一次准确率无法支撑"模型 A 比 B 好"的结论，
       所以用 5 折 × 10 次重复 = 50 个折，报告均值、标准差与 95% 置信区间；
    2. 两两比较必须配对：同一折上同时评估两个模型，再做 Wilcoxon 符号秩检验；
    3. 没有基线的对比没有意义，所以从"多数类"和"关键词规则"逐级往上加，
       每一步的增益都能归因到具体的设计选择；
    4. 性能与部署成本是两个问题，分开测：前者用 macro-F1，后者用模型体积与推理耗时。
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from memo_classifier import char_ngrams, load_corpus, normalize

OUT_DIR = Path(__file__).resolve().parent / "artifacts_compare"
SEED = 42
N_SPLITS = 5
N_REPEATS = 10
BASELINE_NAME = "字符 1-2 元 TF-IDF + NB"

# 手写关键词规则：模拟"不用机器学习"的传统做法，作为最低限度的对照
KEYWORDS: dict[str, tuple[str, ...]] = {
    "工作": ("会议", "汇报", "报表", "客户", "项目", "代码", "bug", "方案", "预算", "审批", "合同", "周报", "排期"),
    "生活": ("超市", "快递", "电费", "理发", "感冒药", "洗车", "露营", "火锅", "牛奶", "猫粮", "物业", "干洗", "宽带"),
    "学习": ("复习", "单词", "论文", "作业", "考试", "实验", "上课", "刷题", "真题", "考研", "讲座", "笔记"),
}


def _make_analyzer(lo: int, hi: int, drop_fillers: bool = True):
    def analyze(text: str) -> list[str]:
        return char_ngrams(normalize(text, drop_fillers), lo, hi)

    return analyze


def build_configs():
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    def tfidf(lo, hi):
        return TfidfVectorizer(
            analyzer=_make_analyzer(lo, hi), sublinear_tf=True, norm="l2", lowercase=False
        )

    return [
        # 下界也要把它包进 Pipeline：DummyClassifier 直接吃字符串列表时，
        # 不同版本的 check_array 行为不一致，先转成稀疏特征最稳。
        (
            "多数类（下界）",
            Pipeline([("f", TfidfVectorizer(analyzer=_make_analyzer(1, 2), lowercase=False)), ("c", DummyClassifier(strategy="most_frequent"))]),
        ),
        ("关键词规则（无模型）", KeywordRules()),
        (BASELINE_NAME, Pipeline([("f", tfidf(1, 2)), ("c", MultinomialNB(alpha=0.5))])),
        ("字符 1-3 元 TF-IDF + NB", Pipeline([("f", tfidf(1, 3)), ("c", MultinomialNB(alpha=0.5))])),
        (
            "计数特征（无 idf）+ NB",
            Pipeline(
                [
                    ("f", CountVectorizer(analyzer=_make_analyzer(1, 2), lowercase=False)),
                    ("c", MultinomialNB(alpha=0.5)),
                ]
            ),
        ),
        ("字符 1-2 元 TF-IDF + LinearSVC", Pipeline([("f", tfidf(1, 2)), ("c", LinearSVC(C=1.0, random_state=SEED))])),
        (
            "字符 1-2 元 TF-IDF + 逻辑回归",
            Pipeline(
                [
                    ("f", tfidf(1, 2)),
                    ("c", LogisticRegression(C=5.0, max_iter=2000, random_state=SEED)),
                ]
            ),
        ),
    ]


class KeywordRules(ClassifierMixin, BaseEstimator):
    """手写规则分类器：命中关键词最多者胜，全不命中退回多数类。

    继承 BaseEstimator 是必需的：cross_validate 会用 clone() 复制估计器，
    没有 get_params 的普通类会直接抛 TypeError。
    """

    def fit(self, texts, labels):
        from collections import Counter

        self.classes_ = np.unique(labels)
        self.default_ = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return np.array([self._predict_one(text) for text in texts])

    def _predict_one(self, text: str) -> str:
        best_label, best_hits = self.default_, 0
        for label, words in KEYWORDS.items():
            hits = sum(1 for word in words if word in text)
            if hits > best_hits:
                best_label, best_hits = label, hits
        return best_label


def t_interval(mean: float, sd: float, n: int) -> tuple[float, float]:
    from scipy import stats

    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return mean - half, mean + half


def evaluate(configs, texts, labels):
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    rows = []
    for name, estimator in configs:
        scores = cross_validate(
            estimator, texts, labels, cv=splitter, scoring=["accuracy", "f1_macro"], n_jobs=1
        )
        folds = np.asarray(scores["test_f1_macro"], dtype=float)
        mean, sd = float(folds.mean()), float(folds.std(ddof=1))
        low, high = t_interval(mean, sd, len(folds))
        rows.append(
            {
                "name": name,
                "f1_mean": mean,
                "f1_sd": sd,
                "ci_low": low,
                "ci_high": high,
                "acc_mean": float(np.mean(scores["test_accuracy"])),
                "folds": folds,
            }
        )
    return rows


def paired_test(rows):
    """以 1-2 元 TF-IDF + NB 为基准，对每个配置做同一批折上的 Wilcoxon 符号秩检验。"""
    from scipy import stats

    baseline = next(row for row in rows if row["name"] == BASELINE_NAME)
    for row in rows:
        if row["name"] == BASELINE_NAME:
            row["p_value"] = float("nan")
            row["note"] = "基线"
            continue
        diff = row["folds"] - baseline["folds"]
        if np.allclose(diff, 0):
            row["p_value"] = 1.0
            row["note"] = "与基线逐折完全一致"
            continue
        try:
            row["p_value"] = float(stats.wilcoxon(row["folds"], baseline["folds"]).pvalue)
        except ValueError:
            row["p_value"] = 1.0
        row["note"] = "显著" if row["p_value"] < 0.05 else "不显著"
    return rows


def learning_curve(texts, labels):
    from sklearn.model_selection import StratifiedKFold, learning_curve

    configs = dict(build_configs())
    sizes = [24, 48, 72, 96, 120]
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    curves = {}
    for name in (BASELINE_NAME, "字符 1-2 元 TF-IDF + LinearSVC"):
        result = learning_curve(
            configs[name],
            texts,
            labels,
            train_sizes=sizes,
            cv=splitter,
            scoring="f1_macro",
            shuffle=True,
            random_state=SEED,
            n_jobs=1,
        )
        curves[name] = (result[0], result[2].mean(axis=1), result[2].std(axis=1))
    return curves


def collect_errors(texts, labels):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    configs = dict(build_configs())
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    predicted = cross_val_predict(configs[BASELINE_NAME], texts, labels, cv=splitter, n_jobs=1)
    return [
        {"text": text, "true": true, "pred": pred}
        for text, true, pred in zip(texts, labels, predicted)
        if true != pred
    ]


def deployment(texts, labels):
    """模型体积与推理耗时的相对比较（同一台机器、同一批样本）。"""
    configs = dict(build_configs())
    nb_pipe = configs[BASELINE_NAME]
    svm_pipe = configs["字符 1-2 元 TF-IDF + LinearSVC"]
    nb_pipe.fit(texts, labels)
    svm_pipe.fit(texts, labels)

    vectorizer = nb_pipe.named_steps["f"]
    nb = nb_pipe.named_steps["c"]
    payload = {
        "features": list(vectorizer.get_feature_names_out()),
        "idf": [round(float(v), 6) for v in vectorizer.idf_],
        "prior": [round(float(v), 6) for v in nb.class_log_prior_],
        "log_prob": [[round(float(v), 6) for v in row] for row in nb.feature_log_prob_],
    }
    nb_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    svm = svm_pipe.named_steps["c"]
    svm_weights = int(svm.coef_.size)
    svm_bytes = svm_weights * 4  # float32

    def per_item_us(estimator, rounds=30):
        estimator.predict(texts)  # 预热
        start = time.perf_counter()
        for _ in range(rounds):
            estimator.predict(texts)
        elapsed = time.perf_counter() - start
        return elapsed / (rounds * len(texts)) * 1e6

    return {
        "nb_bytes": nb_bytes,
        "svm_weights": svm_weights,
        "svm_bytes": svm_bytes,
        "nb_us": per_item_us(nb_pipe),
        "svm_us": per_item_us(svm_pipe),
        "n_features": len(vectorizer.get_feature_names_out()),
    }


def write_outputs(rows, curves, errors, deploy, texts):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    header = "| 对比配置 | macro-F1（均值±标准差） | 95% 置信区间 | 与基线配对检验 |\n|---|---|---|---|\n"
    lines = [header]
    for row in sorted(rows, key=lambda r: r["f1_mean"], reverse=True):
        if row["name"] == BASELINE_NAME:
            test_cell = "基线"
        else:
            test_cell = f"p = {row['p_value']:.3f}（{row['note']}）"
        lines.append(
            f"| {row['name']} | {row['f1_mean']:.3f} ± {row['f1_sd']:.3f} "
            f"| {row['ci_low']:.3f} ～ {row['ci_high']:.3f} | {test_cell} |\n"
        )
    (OUT_DIR / "comparison.md").write_text("".join(lines), encoding="utf-8")

    with (OUT_DIR / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["配置", "macro-F1均值", "标准差", "CI下限", "CI上限", "accuracy均值", "p值", "说明"])
        for row in sorted(rows, key=lambda r: r["f1_mean"], reverse=True):
            writer.writerow(
                [
                    row["name"],
                    f"{row['f1_mean']:.4f}",
                    f"{row['f1_sd']:.4f}",
                    f"{row['ci_low']:.4f}",
                    f"{row['ci_high']:.4f}",
                    f"{row['acc_mean']:.4f}",
                    "" if np.isnan(row["p_value"]) else f"{row['p_value']:.4f}",
                    row["note"],
                ]
            )

    with (OUT_DIR / "errors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["文本", "真实标签", "预测标签"])
        for error in errors:
            writer.writerow([error["text"], error["true"], error["pred"]])

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import font_manager, pyplot as plt

        available = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
            if candidate in available:
                plt.rcParams["font.sans-serif"] = [candidate]
                break
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
        for name, (sizes, mean, sd) in curves.items():
            label = "朴素贝叶斯" if "NB" in name else "线性 SVM"
            ax.plot(sizes, mean, marker="o", label=label)
            ax.fill_between(sizes, mean - sd, mean + sd, alpha=0.15)
        ax.set_xlabel("训练样本数")
        ax.set_ylabel("macro-F1")
        ax.set_title(f"学习曲线（{N_SPLITS} 折，{len(texts)} 条语料）")
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "learning_curve.png")
        plt.close(fig)
    except ImportError:
        print("[warn] 没有 matplotlib，跳过学习曲线图")

    return OUT_DIR


def main() -> int:
    texts, labels = load_corpus()
    print(f"语料 {len(texts)} 条，{N_SPLITS} 折 × {N_REPEATS} 次重复 = {N_SPLITS * N_REPEATS} 个折\n")

    rows = paired_test(evaluate(build_configs(), texts, labels))
    print("| 对比配置 | macro-F1 | 95% CI | p 值 |")
    print("|---|---|---|---|")
    for row in sorted(rows, key=lambda r: r["f1_mean"], reverse=True):
        p = "—" if np.isnan(row["p_value"]) else f"{row['p_value']:.4f}"
        print(f"| {row['name']} | {row['f1_mean']:.3f}±{row['f1_sd']:.3f} | {row['ci_low']:.3f}～{row['ci_high']:.3f} | {p} |")

    curves = learning_curve(texts, labels)
    print("\n学习曲线（macro-F1 均值）")
    for name, (sizes, mean, _) in curves.items():
        points = ", ".join(f"{int(s)}→{m:.3f}" for s, m in zip(sizes, mean))
        print(f"  {'NB' if 'NB' in name else 'SVM'}: {points}")

    errors = collect_errors(texts, labels)
    print(f"\n折外预测错误 {len(errors)}/{len(texts)} 条：")
    for error in errors:
        print(f"  {error['true']} → {error['pred']}：{error['text']}")

    deploy = deployment(texts, labels)
    print("\n部署成本（同一台机器，{0} 条样本）".format(len(texts)))
    print(f"  特征数：{deploy['n_features']}")
    print(f"  NB  JSON：{deploy['nb_bytes'] / 1024:.1f} KB；单条推理 {deploy['nb_us']:.2f} µs")
    print(f"  SVM 权重：{deploy['svm_weights']} 个（float32 约 {deploy['svm_bytes'] / 1024:.1f} KB）；单条推理 {deploy['svm_us']:.2f} µs")

    for row in rows:
        if row["name"] == BASELINE_NAME:
            print(f"\n基线 {BASELINE_NAME}：accuracy={row['acc_mean']:.3f}，macro-F1={row['f1_mean']:.3f}")

    path = write_outputs(rows, curves, errors, deploy, texts)
    print(f"\n产物目录：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
