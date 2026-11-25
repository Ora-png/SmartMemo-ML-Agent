import json
import os
import sys


def load_model():
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 因为 model.json 就在根目录，所以直接拼接文件名即可
    model_path = os.path.join(base_dir, 'model.json')

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}，请确保 model.json 与 predict.py 在同一目录下。")
        sys.exit(1)

    with open(model_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_features(text):
    # 与端侧 ArkTS 逻辑完全一致的 1-gram 和 2-gram 切分
    features = []
    for i in range(len(text)):
        features.append(text[i])
    for i in range(len(text) - 1):
        features.append(text[i:i + 2])
    return features


def predict(text, model):
    classes = model['classes']
    class_log_prior = model['class_log_prior']
    vocabulary = {word: idx for idx, word in enumerate(model['vocabulary'])}
    log_probs = model['log_probs']

    features = extract_features(text)
    scores = class_log_prior.copy()

    for f in features:
        if f in vocabulary:
            idx = vocabulary[f]
            for i in range(len(classes)):
                scores[i] += log_probs[i][idx]

    # 找到得分最高的类别
    max_score = float('-inf')
    best_class = classes[0]
    for i in range(len(scores)):
        if scores[i] > max_score:
            max_score = scores[i]
            best_class = classes[i]

    return best_class


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python predict.py \"你的备忘文本\"")
        sys.exit(1)

    input_text = sys.argv[1]
    model_data = load_model()
    result = predict(input_text, model_data)

    print(f"输入文本: {input_text}")
    print(f"预测分类: {result}")