import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import visualize_classifier

def load_data(file_path):
    return np.loadtxt(file_path, delimiter=',')

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    return y_pred, accuracy, precision, recall, f1

def cross_validate_model(model, X, y, num_folds=3):
    accuracy_cv = cross_val_score(model, X, y, scoring='accuracy', cv=num_folds).mean() * 100
    precision_cv = cross_val_score(model, X, y, scoring='precision_weighted', cv=num_folds).mean() * 100
    recall_cv = cross_val_score(model, X, y, scoring='recall_weighted', cv=num_folds).mean() * 100
    f1_cv = cross_val_score(model, X, y, scoring='f1_weighted', cv=num_folds).mean() * 100
    return accuracy_cv, precision_cv, recall_cv, f1_cv

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def main():
    input_file = 'data_multivar_nb.txt'
    data = load_data(input_file)
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    gnb = GaussianNB()
    svm_classifier = SVC(kernel='linear', random_state=3)
    y_pred_nb, accuracy_nb, precision_nb, recall_nb, f1_nb = evaluate_model(gnb, X_train, y_train, X_test, y_test)
    print("### Результати Наївного Байєсовського Класифікатора ###")
    print(f"Точність: {accuracy_nb:.2f}%")
    print(f"Точність (Precision): {precision_nb:.2f}%")
    print(f"Повнота (Recall): {recall_nb:.2f}%")
    print(f"F1-міра: {f1_nb:.2f}%\n")
    visualize_classifier(gnb, X_test, y_test)
    y_pred_svm, accuracy_svm, precision_svm, recall_svm, f1_svm = evaluate_model(svm_classifier, X_train, y_train, X_test, y_test)
    print("### Результати Класифікатора Опорних Векторів ###")
    print(f"Точність: {accuracy_svm:.2f}%")
    print(f"Точність (Precision): {precision_svm:.2f}%")
    print(f"Повнота (Recall): {recall_svm:.2f}%")
    print(f"F1-міра: {f1_svm:.2f}%\n")
    visualize_classifier(svm_classifier, X_test, y_test)
    accuracy_cv_nb, precision_cv_nb, recall_cv_nb, f1_cv_nb = cross_validate_model(gnb, X, y)
    print("### Перехресна Валідація Наївного Байєсовського Класифікатора ###")
    print(f"Середня точність: {accuracy_cv_nb:.2f}%")
    print(f"Середня точність (Precision): {precision_cv_nb:.2f}%")
    print(f"Середня повнота (Recall): {recall_cv_nb:.2f}%")
    print(f"Середня F1-міра: {f1_cv_nb:.2f}%\n")
    accuracy_cv_svm, precision_cv_svm, recall_cv_svm, f1_cv_svm = cross_validate_model(svm_classifier, X, y)
    print("### Перехресна Валідація Класифікатора Опорних Векторів ###")
    print(f"Середня точність: {accuracy_cv_svm:.2f}%")
    print(f"Середня точність (Precision): {precision_cv_svm:.2f}%")
    print(f"Середня повнота (Recall): {recall_cv_svm:.2f}%")
    print(f"Середня F1-міра: {f1_cv_svm:.2f}%")

if __name__ == "__main__":
    main()
