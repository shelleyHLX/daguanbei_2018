# coding: utf-8

import numpy as np
import time
# from evaluator import cail_evaluator
from multiprocessing import  Pool

scores_path = '../scores/'
scores_names = ['wd_GRU_Attention_3_pdSQ200', 'wd_TextCNN_pdQS200', 'wd_HCNN_pdQS200']

# X: list, [[3.2, 2.2, 6.54], ...]
def sigmoid(X):
    sig = [1.0 / float(1.0 + np.exp(-i)) for i in X]
    return sig

def to_categorical_single_class(cl):
    y = np.zeros(183)
    for i in range(len(cl)):
        y[cl[i]] = 1
    return y

def cail_evaluator_least(predict_labels_list, marked_labels_list):
    # predict labels category
    predict_labels_category = []
    samples = len(predict_labels_list)
    print('num of samples: ', samples)
    for i in range(samples):  # number of samples
        predict_norm = sigmoid(predict_labels_list[i])
        # print(predict_norm)
        predict_category = [1 if i > 0.5 else 0 for i in predict_norm]
        if max(predict_category) == 0:
            predict_category = [1 if i == max(predict_norm) else 0 for i in predict_norm]
        # print(predict_category)
        predict_labels_category.append(predict_category)

    # marked labels category
    marked_labels_category = []
    num_class = len(predict_labels_category[0])
    print('num of classes: ', num_class)
    for i in range(samples):
        marked_category = to_categorical_single_class(marked_labels_list[i])
        marked_labels_category.append(marked_category)

    # print('marked_labels_category', marked_labels_category)
    # print('predict_labels_category', predict_labels_category)
    tp_list = []
    fp_list = []
    fn_list = []
    f1_list = []
    for i in range(num_class):  # 类别个数
        tp = 0.0  # predict=1, truth=1
        fp = 0.0  # predict=1, truth=0
        fn = 0.0  # predict=0, truth=1
        # 样本个数
        pre = [p[i] for p in predict_labels_category]
        mar = [p[i] for p in marked_labels_category]
        pre = np.asarray(pre)
        mar = np.asarray(mar)
        # print('pre: ', pre.shape)
        # print('mar: ', mar.shape)
        # print('marked_labels_category', marked_labels_category)
        # print('pre', pre)
        # print('mar', mar)
        for i in range(len(pre)):
            if pre[i] == 1 and mar[i] == 1:
                tp += 1
            elif pre[i] == 1 and mar[i] == 0:
                fp += 1
            elif pre[i] == 0 and mar[i] == 1:
                fn += 1
        precision = 0.0
        if tp + fp > 0:
            precision = tp / (tp + fp)
        recall = 0.0
        if tp + fn > 0:
            recall = tp / (tp + fn)
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        f1_list.append(f1)

    # micro level
    f1_micro = 0.0
    if sum(tp_list) + sum(fp_list) > 0:
        f1_micro = sum(tp_list) / (sum(tp_list) + sum(fp_list))
    # macro level
    f1_macro = sum(f1_list) / len(f1_list)
    score12 = (f1_macro + f1_micro) / 2.0
    # print(score12)

    return f1_micro, f1_macro, score12

def get_update_weight(score_name):
    """根据线下验证集的 f1 值变化趋势来调整模型的融合权重。
    Args:
        score_name: 需要调整的模型。
    Returns:
        lr: 模型的权重变化。
    """
    global lr  # 权重调整率
    global sum_scores
    score_pre = np.load(scores_path + score_name + '/' + 'predict.npy')
    new_score = sum_scores + score_pre * lr
    f1_micro, f1_macro, score12 = cail_evaluator_least(new_score, marked_label)
    if score12 > last_score12:
        return lr
    else:
        new_score = sum_scores - score_pre * lr
        f1_micro, f1_macro, score12 = cail_evaluator_least(new_score, marked_label)
        if score12 > last_score12:
            return -lr
    return 0.0

time0 = time.time()
# init
predict_label = np.load(scores_path + scores_names[0] + '/' + 'predict.npy')
marked_label = np.load(scores_path + scores_names[0] + '/' + 'origin.npy')
f1_micro, f1_macro, score12 = cail_evaluator_least(predict_label, marked_label)

sum_scores = predict_label
last_score12 = score12
best_score12 = score12
lr = 0.15

# 更新权重
score12_list = list()
w_list = list()
decay1 = 0.995
decay2 = 0.95
decay = decay1
weights = np.random.uniform(-1, 1, size=len(scores_names))

for i in range(200):
    if i == 50:
        decay = decay2    # 增加下降速度
    lr = lr * decay
    p = Pool(8)
    weights = np.asarray(weights)
    print('=='*10, "i=", i, ',', "lr=", lr)
    print('LAST_score12=', last_score12)
    update_w = list(p.map(get_update_weight, scores_names))
    update_w = np.asarray(update_w)
    p.close()
    p.join()
    print('update_w=', update_w)
    weights = weights + update_w  # 更新
    print('new_w=', weights)
    sum_scores = np.zeros((len(marked_label), 183), dtype=float)
    print('sum_scores: ', sum_scores.shape)
    for i in range(len(weights)):       # 新的权重组合
        scores_name = scores_names[i]
        score = np.load(scores_path + scores_name + "/" + 'predict.npy')
        sum_scores = sum_scores + score * weights[i]     # 新的 sum_scores
    f1_micro, f1_macro, score12 = cail_evaluator_least(sum_scores, marked_label)
    print('NEW_score12=', score12)
    if score12 > best_score12:
        best_score12 = score12
        np.save('best_weights.npy', weights)
    score12_list.append(score12)
    w_list.append(weights)
    last_score12 = score12  # 更新 score12
    print('**Best_f1=%f; Speed: %g s / epoch.' % (best_score12, time.time() - time0))
    time0 = time.time()

