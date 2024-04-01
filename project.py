import pandas
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from warnings import simplefilter
import warnings
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



## Data Preprocess
df = pd.read_csv('bank-full.csv', sep=';')
print(df.head().T)  # 查看特征
print(df.info())  # 没有缺失值

# print(df['job'].unique())
df[['default']] = df[['default']].replace(['no', 'yes'], [0, 1])
df[['housing']] = df[['housing']].replace(['yes', 'no'], [1, 0])
df[['loan']] = df[['loan']].replace(['no', 'yes'], [0, 1])
df[['month']] = df[['month']].replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
df[['y']] = df[['y']].replace(['no', 'yes'], [0, 1])

# categorical data -- one hot encoding
df = pd.get_dummies(df)
df = df.astype('int')
print(df.head().T)
X = df.drop('y', axis=1)
y = df['y']
print(X.head().T)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Selector of Original Skewed Dataset or Balanced Dataset
print("--------Using Original Skewed Dataset: Please enter '0'---------",
      "\n--------Using Balanced Dataset by SMOTETomek: Please enter '1'---------\n")
try:
    choice=int(input('Please enter a number:'))
    #输入的判断
    if choice==0:
        print('\nUsing Original Skewed Dataset')
        print(Counter(y_train))  # 查看数据分布
    else:
        print('\nUsing Balanced Dataset')
        print('----------------Resampling----------------')
        print(Counter(y_train))  # 查看数据分布
        smote = SMOTETomek(random_state=42)  # 过采样
        smote_X, smote_y = smote.fit_resample(X_train, y_train)
        print(Counter(smote_y))
        X_train = smote_X
        y_train = smote_y
except ValueError:
    print('\nEnter wrong number. Please rerun the program')



## Supervised Learning
def supervised_model(X_train, X_test, y_train, y_test):
    t = time.time()
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)  # accuracy
    print("Accuracy Score of Supervised Learning is:", acc)
    f = f1_score(y_test, y_pred, average='micro', zero_division=1)  # f1
    print("F1 Score of Supervised Learning is:", f)
    run_time = time.time() - t  # runtime
    print("Run time of Supervised Learning is:", run_time,'\n')

    return y_pred

print('\n----------------Supervised learning----------------')
supervised_learning = supervised_model(X_train, X_test, y_train, y_test)



## Self-training Algorithm
def self_training(X_train, X_test, y_train, y_test, n_unlabelled):
    t = time.time()
    y_train_self = y_train.copy()
    rng = np.random.RandomState(42)  # 伪随机数生成器：用确定性的算法计算出来的似来自[0,1]均匀分布的随机数序列
    random_unlabeled_points = rng.rand(y_train_self.shape[0]) < n_unlabelled  # 生成的y_train个伪随机数，小于0.3为Ture；大于0.3为False；30% unlabelled
    y_train_self[random_unlabeled_points] = -1  # 小于0.3的Ture的；未知的label设置为-1
    print('The level of unlabelled data is', n_unlabelled)
    # print(y_train.value_counts())  # 查看y_train数据分布

    # self_model = make_pipeline(CalibratedClassifierCV(LinearSVC()))  # to speed up the svc model
    self_model = SVC(probability=True)
    # self_model = DecisionTreeClassifier()
    self_training_model = SelfTrainingClassifier(self_model)
    self_training_model.fit(X_train, y_train_self)
    y_pred = self_training_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)  # accuracy
    print("Accuracy Score of Self-training is:", acc)
    f = f1_score(y_test, y_pred, average='micro', zero_division=1)  # f1
    print("F1 Score of Self-training is:", f)
    run_time = time.time() - t  # run runtime
    print("Run time of Self-training is:", run_time,'\n')

    return y_pred

print('\n----------------Self-training----------------')
self_50 = self_training(X_train, X_test, y_train, y_test, 0.5)  #  50% unlabelled data
self_75 = self_training(X_train, X_test, y_train, y_test, 0.75)  #  75% unlabelled data
self_95 = self_training(X_train, X_test, y_train, y_test, 0.95)  #  90% unlabelled data
self_99 = self_training(X_train, X_test, y_train, y_test, 0.99)  #  99% unlabelled data




## Co-training Algorithm
def co_training(X_train, X_test, y_train, y_test, num_iterations, n_unlabelled):
    t = time.time()
    print('The level of unlabelled data is', n_unlabelled)
    X_labelled, X_unlabelled, y_labelled, y_unlabelled = train_test_split(X_train, y_train, test_size=n_unlabelled, random_state=42)
    X_view1, X_view2, y_view1, y_view2 = train_test_split(X_labelled, y_labelled, test_size=0.5, random_state=42)

    for i in range(num_iterations):
        # 训练分类器1
        clf1 = SVC()
        clf1.fit(X_view1, y_view1)

        # 训练分类器2
        clf2 = SVC()
        clf2.fit(X_view2, y_view2)

        # 使用分类器1和分类器2对未标记样本进行预测
        y_pred_view1 = clf1.predict(X_unlabelled)
        y_pred_view2 = clf2.predict(X_unlabelled)

        # 将高置信度预测样本添加到标记样本中
        X_view1 = np.concatenate((X_view1, X_unlabelled[y_pred_view2 == y_unlabelled]))
        y_view1 = np.concatenate((y_view1, y_unlabelled[y_pred_view2 == y_unlabelled]))
        X_view2 = np.concatenate((X_view2, X_unlabelled[y_pred_view1 == y_unlabelled]))
        y_view2 = np.concatenate((y_view2, y_unlabelled[y_pred_view1 == y_unlabelled]))

        # 将添加的样本从无标签样本中移除
        remove_unlabelled = pd.concat((X_unlabelled[y_pred_view2 == y_unlabelled], X_unlabelled[y_pred_view1 == y_unlabelled]), axis=0)
        remove_unlabelled = remove_unlabelled.drop_duplicates()
        X_unlabelled = X_unlabelled.drop(index=remove_unlabelled.index, axis=0)
        y_unlabelled = y_unlabelled.drop(index=remove_unlabelled.index, axis=0)


    # 将视图1和视图2合并为完整的训练集
    X_train_co = np.concatenate((X_view1, X_view2))
    y_train_co = np.concatenate((y_view1, y_view2))


    # 在完整的训练集上训练最终的分类器
    clf_final = SVC()
    clf_final.fit(X_train_co, y_train_co)
    y_pred = clf_final.predict(X_test)

    acc = accuracy_score(y_test, y_pred)  # accuracy
    print("Accuracy Score of Co-training is:", acc)
    f = f1_score(y_test, y_pred, average='micro', zero_division=1)  # f1
    print("F1 Score of Co-training is:", f)
    run_time = time.time() - t  # run runtime
    print("Run time of Co-training is:", run_time, '\n')

    return y_pred

print('\n----------------Co-training----------------')
co_50 = co_training(X_train, X_test, y_train, y_test, num_iterations=2, n_unlabelled=0.5)
co_75 = co_training(X_train, X_test, y_train, y_test, num_iterations=2, n_unlabelled=0.75)
co_95 = co_training(X_train, X_test, y_train, y_test, num_iterations=2, n_unlabelled=0.90)
co_99 = co_training(X_train, X_test, y_train, y_test, num_iterations=2, n_unlabelled=0.99)




## Semi-supervised Ensemble
def semi_boosting(X_train, y_train, X_test, y_test, num_iterations, n_unlabelled):
    t = time.time()
    print('The level of unlabelled data is', n_unlabelled)
    X_labelled, X_unlabelled, y_labelled, y_unlabelled = train_test_split(X_train, y_train, test_size=n_unlabelled, random_state=42)
    y_unlabelled[0:] = -1  # -1 indicates unlabeled

    weights_labelled = np.ones(len(X_labelled)) / len(X_labelled)
    weights_unlabelled = np.ones(len(X_unlabelled)) / len(X_unlabelled)
    weights_train = weights_labelled

    for j in range(num_iterations):
        weak_learner = DecisionTreeClassifier(max_depth=5)
        weak_learner.fit(X_labelled, y_labelled, sample_weight=weights_train)
        weak_learner_pred = weak_learner.predict(X_labelled)

        # Update labelled weights
        errors = np.abs(weak_learner_pred - y_labelled)
        error_rate = np.sum(errors * weights_labelled) / np.sum(weights_labelled)
        beta = error_rate / (1 - error_rate)
        weights_labelled *= np.power(beta, 1 - errors)

        pseudo_labels = weak_learner.predict(X_unlabelled)

        # Update unlabelled weights
        weights_unlabelled *= np.exp(-beta * pseudo_labels)
        weights_unlabelled /= np.sum(weights_unlabelled)

        # Add high confident pseudo-labeled
        X_labelled = np.concatenate((X_labelled, X_unlabelled[pseudo_labels == y_unlabelled]))
        y_labelled = np.concatenate((y_labelled, y_unlabelled[pseudo_labels == y_unlabelled]))
        weights_train = np.concatenate((weights_labelled, weights_unlabelled[pseudo_labels == y_unlabelled]))

        remove_unlabelled = X_unlabelled[pseudo_labels == y_unlabelled]
        X_unlabelled = X_unlabelled.drop(index=remove_unlabelled.index, axis=0)
        y_unlabelled = y_unlabelled.drop(index=remove_unlabelled.index, axis=0)
        print(weights_train)

    final_model = GradientBoostingClassifier(max_depth=5, n_estimators=num_iterations)
    final_model.fit(X_labelled, y_labelled, sample_weight=weights_train)
    y_pred = final_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)  # accuracy
    print("Accuracy Score of Self-training is:", acc)
    f = f1_score(y_test, y_pred, average='micro', zero_division=1)  # f1
    print("F1 Score of Self-training is:", f)
    run_time = time.time() - t  # run runtime
    print("Run time of Self-training is:", run_time,'\n')

    return y_pred

print('\n----------------Semi-supervised Ensemble----------------')
boost_50 = semi_boosting(X_train, y_train, X_test, y_test, 5, 0.5)
boost_75 = semi_boosting(X_train, y_train, X_test, y_test, 5, 0.75)
boost_90 = semi_boosting(X_train, y_train, X_test, y_test, 5, 0.90)
boost_99 = semi_boosting(X_train, y_train, X_test, y_test, 5, 0.99)




## Unsupervised Pretraining / Intrinsically Semi-supervised Learning






## Draw ROC Curve
def plot_roc_curve(y_test, y_pred_list, labels):
    plt.figure(figsize=(8, 6))
    for i in range(len(y_pred_list)):
        fpr, tpr, _ = roc_curve(y_test, y_pred_list[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (labels[i], roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Call the function to plot ROC curves for all models
y_pred_list = []  # List to store predicted probabilities for each model
labels = []
# print(Counter(supervised_learning))
y_pred_list = [supervised_learning,
               self_50, self_75, self_95, self_99,
               co_50, co_75, co_95, co_99,
               boost_50, boost_75, boost_90, boost_99]
labels = ['Supervised Model',
          'Self_50', 'Self_75', 'Self_95', 'Self_99',
          'Co_50', 'Co_75', 'Co_95', 'Co_99',
          'Boost_50', 'Boost_75', 'Boost_90', 'Boost_99']
plot_roc_curve(y_test, y_pred_list, labels)
