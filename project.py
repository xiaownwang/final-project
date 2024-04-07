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
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from warnings import simplefilter
import warnings
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



## Data Preprocess
df = pd.read_csv('bank-full.csv', sep=';')
print(df.head().T)
print(df.info())  # no default value

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
    # input a number
    if choice==0:
        print('\nUsing Original Skewed Dataset')
        print(Counter(y_train)) 
    else:
        print('\nUsing Balanced Dataset')
        print('----------------Resampling----------------')
        print(Counter(y_train))
        smote = SMOTETomek(random_state=42)  # resample
        smote_X, smote_y = smote.fit_resample(X_train, y_train)
        print(Counter(smote_y))
        X_train = smote_X
        y_train = smote_y
except ValueError:
    print('\nEnter wrong number. Please rerun the program')



## Supervised Learning
def supervised_learning(X_train, X_test, y_train, y_test):
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
supervised_learning = supervised_learning(X_train, X_test, y_train, y_test)



## Self-training Algorithm
def self_training(X_train, X_test, y_train, y_test, n_unlabelled):
    t = time.time()
    y_train_self = y_train.copy()
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(y_train_self.shape[0]) < n_unlabelled
    y_train_self[random_unlabeled_points] = -1
    print('The level of unlabelled data is', n_unlabelled)
    # print(y_train.value_counts()) 

    # self_model = make_pipeline(CalibratedClassifierCV(LinearSVC()))  # to speed up the svc model
    self_model = SVC(probability=True)
    # self_model = DecisionTreeClassifier()
    self_training_model = SelfTrainingClassifier(self_model, threshold=0.95)
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
    X_classifier1, X_classifier2 = X_labelled.iloc[:, 0:16], X_labelled.iloc[:, 16:]
    y_classifier1, y_classifier2 = y_labelled.copy(), y_labelled.copy()
    X_unlabelled1, X_unlabelled2 = X_unlabelled.iloc[:, 0:16], X_unlabelled.iloc[:, 16:]

    clf1 = SVC(probability=True)  # classifier 1
    clf2 = SVC(probability=True)  # classifier 2

    for i in range(num_iterations):
        clf1.fit(X_classifier1, y_classifier1)
        clf2.fit(X_classifier2, y_classifier2)
        pseudo_prob1 = clf1.predict_proba(X_unlabelled1)
        pseudo_prob2 = clf2.predict_proba(X_unlabelled2)

        # high confidence
        threshold = 0.95  # Set confidence threshold
        high_confidence1 = np.max(pseudo_prob1, axis=1) > threshold  # high confidence indices 1
        X_high_confidence1, y_high_confidence1 = X_unlabelled2[high_confidence1], y_unlabelled[high_confidence1]
        high_confidence2 = np.max(pseudo_prob2, axis=1) > threshold  # high confidence indices 2
        X_high_confidence2, y_high_confidence2 = X_unlabelled1[high_confidence2], y_unlabelled[high_confidence2]

        high_confidence_index = np.concatenate((X_high_confidence1.index, X_high_confidence2.index))
        high_confidence_index = np.unique(high_confidence_index)

        # add pseudo labels
        X_classifier1 = np.concatenate((X_classifier1, X_high_confidence2), axis=0)
        y_classifier1 = np.concatenate((y_classifier1, y_high_confidence2), axis=0)
        X_classifier2 = np.concatenate((X_classifier2, X_high_confidence1), axis=0)
        y_classifier2 = np.concatenate((y_classifier2, y_high_confidence1), axis=0)

        X_unlabelled1 = X_unlabelled1.drop(index=high_confidence_index, axis=0)
        X_unlabelled2 = X_unlabelled2.drop(index=high_confidence_index, axis=0)
        y_unlabelled = y_unlabelled.drop(index=high_confidence_index, axis=0)

    y_pred1 = clf1.predict(X_test.iloc[:, 0:16])
    y_pred2 = clf2.predict(X_test.iloc[:, 16:])
    y_pred = np.array(  # ensemble predictions
        [y_pred1[i] if y_pred1[i] == y_pred2[i] else np.random.choice([y_pred1[i], y_pred2[i]]) for i in range(len(y_pred1))])

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
    weights_train = np.ones(len(X_labelled)) / len(X_labelled)
    models = []
    weights = []

    for j in range(num_iterations):
        weak_learner = SVC(probability=True)
        weak_learner.fit(X_labelled, y_labelled, sample_weight=weights_train)
        weak_learner_pred = weak_learner.predict(X_labelled)
        pseudo_prob = weak_learner.predict_proba(X_unlabelled)

        # Update weights
        errors = np.abs(weak_learner_pred - y_labelled)
        error_rate = np.sum(errors * weights_train) / np.sum(weights_train)
        beta = error_rate / (1 - error_rate)
        weights_train *= np.power(beta, 1 - errors)  # reduce the weight of correct predictions
        weight_basemodel = 0.25 * np.log((1 - error_rate) / error_rate)

        models.append(weak_learner)  # Save model
        weights.append(weight_basemodel)  # save weights

        # high confidence
        threshold = 0.95  # Set confidence threshold
        high_confidence = np.max(pseudo_prob, axis=1) > threshold  # high confidence indices
        X_high_confidence, y_high_confidence = X_unlabelled[high_confidence], y_unlabelled[high_confidence]

        # add pseudo lables
        X_labelled = np.concatenate((X_labelled, X_high_confidence), axis=0)
        y_labelled = np.concatenate((y_labelled, y_high_confidence), axis=0)
        X_unlabelled = X_unlabelled.drop(index=X_high_confidence.index, axis=0)
        y_unlabelled = y_unlabelled.drop(index=y_high_confidence.index, axis=0)
        weights_train = np.concatenate((weights_train, np.random.choice(range(0, 1), size=y_high_confidence.shape[0])))

    # Predict
    y_pred = np.zeros(X_test.shape[0])
    # Predict weighting each model
    for i in range(len(models)):
        y_pred += weights[i] * models[i].predict(X_test)
    y_pred = np.array([1 if y_pred[x] > 0.5 else 0 for x in range(len(y_pred))])

    acc = accuracy_score(y_test, y_pred)  # accuracy
    print("Accuracy Score of Semi-boosting Ensemble is:", acc)
    f = f1_score(y_test, y_pred, average='micro', zero_division=1)  # f1
    print("F1 Score of Semi-boosting Ensemble is:", f)
    run_time = time.time() - t  # run runtime
    print("Run time of Semi-boosting Ensemble is:", run_time,'\n')

    return y_pred

print('\n----------------Semi-supervised Ensemble----------------')
boost_50 = semi_boosting(X_train, y_train, X_test, y_test, 2, 0.5)
boost_75 = semi_boosting(X_train, y_train, X_test, y_test, 2, 0.75)
boost_90 = semi_boosting(X_train, y_train, X_test, y_test, 2, 0.90)
boost_99 = semi_boosting(X_train, y_train, X_test, y_test, 2, 0.99)




## Unsupervised Pretraining / Intrinsically Semi-supervised Learning
def semi_pretraining(X_train, y_train, X_test, y_test, n_unlabelled):
    t = time.time()
    X_labelled, X_unlabelled, y_labelled, y_unlabelled = train_test_split(X_train, y_train, test_size=n_unlabelled, random_state=42)

    # Autoencoder model for unsupervised pre-training
    input_dim = X_unlabelled.shape[1]  # input size
    encoding_dim = 128  # output size
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(int(encoding_dim / 2), activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_unlabelled, X_unlabelled, epochs=10, batch_size=32)

    # Fine-tuning for supervised learning
    encoder_layer = autoencoder.layers[1]
    encoder_layer.trainable = False  # Freeze encoder's layers
    classifier = layers.Dense(64, activation='relu')(encoder_layer.output)
    classifier_output = layers.Dense(1, activation='sigmoid')(classifier)

    # Create the supervised model
    supervised_model = models.Model(inputs=encoder_layer.input, outputs=classifier_output)
    supervised_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    supervised_model.fit(X_labelled, y_labelled, epochs=10, batch_size=32)
    y_pred = np.argmax(supervised_model.predict(X_test), axis=-1)

    test_loss, test_acc = supervised_model.evaluate(X_test, y_test)  # accuracy
    print("Accuracy Score of Unsupervised-pretraining Neural Network is:", test_acc)
    f = f1_score(y_test, y_pred, average='micro', zero_division=1)  # f1
    print("F1 Score of Unsupervised-pretraining Neural Network is:", f)
    run_time = time.time() - t  # run runtime
    print("Run time of Unsupervised-pretraining Neural Network is:", run_time, '\n')

    return y_pred

print('\n----------------Neural Network with Unsupervised Pretraining----------------')
pretrain_50 = semi_pretraining(X_train, y_train, X_test, y_test, 0.5)
pretrain_75 = semi_pretraining(X_train, y_train, X_test, y_test, 0.75)
pretrain_90 = semi_pretraining(X_train, y_train, X_test, y_test, 0.90)
pretrain_99 = semi_pretraining(X_train, y_train, X_test, y_test, 0.99)




## Draw ROC Curve
def plot_roc_curve(y_test, y_pred_list, labels):
    plt.figure(figsize=(8, 6))
    for i in range(len(y_pred_list)):
        fpr, tpr, _ = roc_curve(y_test, y_pred_list[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s (AUC = %0.3f)' % (labels[i], roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# ROC curves for all models
y_pred_list = []  # List to store predicted probabilities for each model
labels = []
# print(Counter(supervised_learning))
y_pred_list = [supervised_learning,
               self_50, self_75, self_95, self_99,
               co_50, co_75, co_95, co_99,
               boost_50, boost_75, boost_90, boost_99,
               pretrain_50, pretrain_75, pretrain_90, pretrain_99]
labels = ['Supervised Learning',
          'Self_50', 'Self_75', 'Self_95', 'Self_99',
          'Co_50', 'Co_75', 'Co_95', 'Co_99',
          'Boost_50', 'Boost_75', 'Boost_90', 'Boost_99',
          'Pretrain_50', 'Pretrain_75', 'Pretrain_90', 'Pretrain_99']
plot_roc_curve(y_test, y_pred_list, labels)

