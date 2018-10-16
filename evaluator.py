# -*- coding:utf-8 -*-

from __future__ import division
import numpy as np


def sigmoid(X):
    sig = [1.0 / float(1.0 + np.exp(-x)) for x in X]
    return sig


def to_categorical_single_class(cl):
    y = np.zeros(19)
    y[cl - 1] = 1
    return y


def cail_evaluator(predict_labels_list, marked_labels_list):
    # predict labels category
    predict_labels_category = []
    samples = len(predict_labels_list)
    print('num of samples: ', samples)
    # pred = codecs.open('pred' + str(samples) + '.txt', 'a', 'utf-8')
    # mark = codecs.open('mark' + str(samples) + '.txt', 'a', 'utf-8')
    for i in range(samples):  # number of samples
        predict_norm = sigmoid(predict_labels_list[i])
        # for e in predict_norm:
        #     e = round(e, 6)
        #     pred.write(str(e) + ' ')
        # pred.write('\n')
        predict_category = [1 if i == max(predict_norm) else 0 for i in predict_norm]
        predict_labels_category.append(predict_category)
    # pred.close()
    # marked labels category
    marked_labels_category = []
    num_class = len(predict_labels_category[0])
    print('num of classes: ', num_class)
    for i in range(samples):
        marked_category = to_categorical_single_class(marked_labels_list[i])
        # for e in marked_labels_list[i]:
        #     mark.write(str(e) + ' ')
        # mark.write('\n')
        marked_labels_category.append(marked_category)
    # mark.close()
    tp_list = []
    fp_list = []
    fn_list = []
    f1_list = []
    for i in range(num_class):  # 类别个数
        # print(i)
        tp = 0.0  # predict=1, truth=1
        fp = 0.0  # predict=1, truth=0
        fn = 0.0  # predict=0, truth=1
        # 样本个数
        pre = [p[i] for p in predict_labels_category]
        mar = [p[i] for p in marked_labels_category]
        pre = np.asarray(pre)
        mar = np.asarray(mar)

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
        # print('tp: %s, fp: %s, fn:%s, f1:%s' %(tp, fp, fn, f1))

    # micro level
    f1_micro = 0.0
    if sum(tp_list) + sum(fp_list) > 0:
        f1_micro = sum(tp_list) / (sum(tp_list) + sum(fp_list))
    # macro level
    f1_macro = sum(f1_list) / len(f1_list)
    score12 = (f1_macro + f1_micro) / 2.0

    return f1_micro, f1_macro, score12


def daguanbei_predict(predict_labels_list):
    samples = len(predict_labels_list)
    predict_labels_category = []

    for i in range(samples):  # number of samples
        predict_norm = sigmoid(predict_labels_list[i])
        # for e in predict_norm:
        #     e = round(e, 6)
        #     pred.write(str(e) + ' ')
        # pred.write('\n')
        predict_category = [1 if i == max(predict_norm) else 0 for i in predict_norm]
        predict_labels_category.append(predict_category)
    labels = []
    for pre in predict_labels_category:
        ix = pre.index(1)
        labels.append(ix + 1)
    return labels


if __name__ == '__main__':
    a = [0, 0, 0, 1, 0, 0]
    ix = a.index(1)
    print(ix)