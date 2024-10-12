import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from utilities import visualize_classifier

df = pd.read_csv('data_metrics.csv')
print(f"\nDataframe:\n{df.head()}")

thresh = 0.5
df['predicted_RF'] = (df.model_RF > thresh).astype(int)
df['predicted_LR'] = (df.model_LR > thresh).astype(int)
print(f"\nDataframe with predictions:\n{df.head()}")

conf_matrix_sklearn = confusion_matrix(df.actual_label.values, df.predicted_RF.values)
print(f"\nConfusion matrix:\n{conf_matrix_sklearn}")

def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def geyna_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

def geyna_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

def geyna_f1_score(y_true, y_pred):
    recall = geyna_recall_score(y_true, y_pred)
    precision = geyna_precision_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall)

def geyna_accuracy(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)

def geyna_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

my_confusion_matrix = geyna_confusion_matrix(df.actual_label.values, df.predicted_RF.values)
print(f"\nMy confusion matrix:\n{my_confusion_matrix}")

my_accuracy = geyna_accuracy(df.actual_label.values, df.predicted_RF.values)
accuracy_sklearn = accuracy_score(df.actual_label.values, df.predicted_RF.values)
my_recall = geyna_recall_score(df.actual_label.values, df.predicted_RF.values)
recall_sklearn = recall_score(df.actual_label.values, df.predicted_RF.values)
my_precision = geyna_precision_score(df.actual_label.values, df.predicted_RF.values)
precision_sklearn = precision_score(df.actual_label.values, df.predicted_RF.values)
my_f1 = geyna_f1_score(df.actual_label.values, df.predicted_RF.values)
f1_sklearn = f1_score(df.actual_label.values, df.predicted_RF.values)

print('Accuracy RF: %.3f'%(geyna_accuracy(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f'%(geyna_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f'%(geyna_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f'%(geyna_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f'%(geyna_accuracy(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f'%(geyna_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f'%(geyna_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f'%(geyna_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC (LR): %.3f'% auc_LR)
print('AUC (RF): %.3f'% auc_RF)
