import numpy as np


def cal_confusion_matrix(y_binary, y_pred):
    # y_binary: tensor, 0 or 1
    # y_pred: tensor, 0~1

    TP = 0  # 11
    FN = 0  # 10
    FP = 0  # 01
    TN = 0  # 00

    for i in range(len(y_binary)):
        # print(y_binary[i], y_pred[i].item())
        # if y_binary[i].item() == 1:
        if y_binary[i] == 1:
            if y_pred[i].item() >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred[i].item() < 0.5:
                TN += 1
            else:
                FP += 1
    print('TP, FN, FP, TN:', TP, FN, FP, TN)

    accuracy = (TP + TN) * 1.0 / (TP + FN + FP + TN)

    if (TP + FN) == 0:
        sensitivity = 0.0
    else:
        sensitivity = TP * 1.0 / (TP + FN)

    if (TP + FP) == 0:
        precision = 0.0
    else:
        precision = TP * 1.0 / (TP + FP)

    if (precision + sensitivity) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * sensitivity / (precision + sensitivity)

    if (TN + FP) == 0:
        specificity = 0.0
    else:
        specificity = TN * 1.0 / (FP + TN)

    balanced_accuracy = 0.5 * (sensitivity + specificity)
    g_balanced_accuracy = (sensitivity * specificity) ** 0.5

    return accuracy, sensitivity, precision, f1_score, specificity, balanced_accuracy, g_balanced_accuracy
