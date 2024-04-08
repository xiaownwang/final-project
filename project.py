import pandas
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
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
print(df.info())  # No default value

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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.values.ravel()



## Selector of Original Skewed Dataset or Balanced Dataset
print("--------Using Original Skewed Dataset: Please enter '0'---------",
      "\n--------Using Balanced Dataset by SMOTETomek: Please enter '1'---------\n")
try:
    choice=int(input('Please enter a number:'))
    # Enter a number
    if choice==0:
        print('\nUsing Original Skewed Dataset')
        print(Counter(y_train))
    else:
        print('\nUsing Balanced Dataset')
        print('----------------Resampling----------------')
        print(Counter(y_train))
        smote = SMOTETomek(random_state=42)
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
    f = f1_score(y_test, y_pred)  # f1
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

    self_model = SVC(probability=True)
    self_training_model = SelfTrainingClassifier(self_model, threshold=0.95)
    self_training_model.fit(X_train, y_train_self)
    y_pred = self_training_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)  # accuracy
    print("Accuracy Score of Self-training is:", acc)
    f = f1_score(y_test, y_pred)  # f1
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
    X_classifier1, X_classifier2 = X_labelled[:, 0:16], X_labelled[:, 16:]
    y_classifier1, y_classifier2 = y_labelled.copy(), y_labelled.copy()
    X_unlabelled1, X_unlabelled2 = X_unlabelled[:, 0:16], X_unlabelled[:, 16:]

    clf1 = SVC(probability=True)  # classifier 1
    clf2 = SVC(probability=True)  # classifier 2

    for i in range(num_iterations):
        clf1.fit(X_classifier1, y_classifier1)
        clf2.fit(X_classifier2, y_classifier2)
        pseudo_prob1 = clf1.predict_proba(X_unlabelled1)
        pseudo_prob2 = clf2.predict_proba(X_unlabelled2)

        # Get high-confidence pseudo labels
        threshold = 0.95  # Set confidence threshold
        high_confidence1 = np.max(pseudo_prob1, axis=1) > threshold  # high confidence indices 1
        X_high_confidence1, y_high_confidence1 = X_unlabelled2[high_confidence1], y_unlabelled[high_confidence1]
        high_confidence2 = np.max(pseudo_prob2, axis=1) > threshold  # high confidence indices 2
        X_high_confidence2, y_high_confidence2 = X_unlabelled1[high_confidence2], y_unlabelled[high_confidence2]



        high_confidence_index = np.concatenate((np.where(high_confidence1)[0], np.where(high_confidence2)[0]))
        high_confidence_index = np.unique(high_confidence_index)

        # Add high-confidence pseudo labels
        X_classifier1 = np.concatenate((X_classifier1, X_high_confidence2), axis=0)
        y_classifier1 = np.concatenate((y_classifier1, y_high_confidence2), axis=0)
        X_classifier2 = np.concatenate((X_classifier2, X_high_confidence1), axis=0)
        y_classifier2 = np.concatenate((y_classifier2, y_high_confidence1), axis=0)

        X_unlabelled1 = np.delete(X_unlabelled1, high_confidence_index, axis=0)
        X_unlabelled2 = np.delete(X_unlabelled2, high_confidence_index, axis=0)
        y_unlabelled = np.delete(y_unlabelled, high_confidence_index, axis=0)

    y_pred1 = clf1.predict(X_test[:, 0:16])
    y_pred2 = clf2.predict(X_test[:, 16:])
    y_pred = np.array(  # ensemble predictions
        [y_pred1[i] if y_pred1[i] == y_pred2[i] else np.random.choice([y_pred1[i], y_pred2[i]]) for i in range(len(y_pred1))])

    acc = accuracy_score(y_test, y_pred)  # accuracy
    print("Accuracy Score of Co-training is:", acc)
    f = f1_score(y_test, y_pred)  # f1
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
def semi_boosting(X_train, y_train, X_test, y_test, n_estimators, n_unlabelled):
    t = time.time()
    print('The level of unlabelled data is', n_unlabelled)
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train, y_train, test_size=n_unlabelled, random_state=42)
    n_samples = X_labeled.shape[0]
    weights = np.ones(n_samples) / n_samples
    models = []
    alphas = []

    for i in range(n_estimators):
        clf = SVC(probability=True)
        clf.fit(X_labeled, y_labeled)
        y_pred = clf.predict(X_labeled)

        misclassified_weight = np.sum(weights * (y_labeled != y_pred))
        epsilon = misclassified_weight / np.sum(weights)  # error rate
        alpha = 0.5 * np.log((1 - epsilon) / epsilon)  # weight of base classifier

        weights = weights * np.exp(-alpha * y_labeled * y_pred)  # update weight
        weights /= np.sum(weights)

        models.append(clf)
        alphas.append(alpha)

        # Get high-confidence pseudo labels
        y_unlabeled_pred = clf.predict(X_unlabeled)
        confidence = np.abs(clf.predict_proba(X_unlabeled).max(axis=1))  # confidence level
        pseudo_mask = confidence > 0.95
        X_pseudo_labeled = X_unlabeled[pseudo_mask]
        y_pseudo_labeled = y_unlabeled_pred[pseudo_mask]
        # Update weight with pseudo labels
        pseudo_weights = np.ones(len(X_pseudo_labeled)) * np.exp(alpha)  # weight: e^alpha
        weights = np.concatenate([weights, pseudo_weights])

        # Add high-confidence pseudo labels
        X_labeled = np.concatenate([X_labeled, X_pseudo_labeled])
        y_labeled = np.concatenate([y_labeled, y_pseudo_labeled])
        X_unlabeled = X_unlabeled[~pseudo_mask]

    y_pred = np.sum([alpha * model.predict(X_test) for alpha, model in zip(alphas, models)], axis=0)
    y_pred = np.sign(y_pred)

    acc = accuracy_score(y_test, y_pred)  # accuracy
    print("Accuracy Score of Semi-boosting Ensemble is:", acc)
    f = f1_score(y_test, y_pred)  # f1
    print("F1 Score of Semi-boosting Ensemble is:", f)
    run_time = time.time() - t  # run runtime
    print("Run time of Semi-boosting Ensemble is:", run_time, '\n')

    return y_pred

print('\n----------------Semi-supervised Ensemble----------------')
boost_50 = semi_boosting(X_train, y_train, X_test, y_test, 2, 0.5)
boost_75 = semi_boosting(X_train, y_train, X_test, y_test, 2, 0.75)
boost_90 = semi_boosting(X_train, y_train, X_test, y_test, 2, 0.90)
boost_99 = semi_boosting(X_train, y_train, X_test, y_test, 2, 0.99)




## Unsupervised Pretraining / Stacked AutoEncoder Learning
def semi_pretraining(X_train, y_train, X_test, y_test, n_unlabelled):
    t = time.time()
    X_labelled, X_unlabelled, y_labelled, y_unlabelled = train_test_split(X_train, y_train, test_size=n_unlabelled, random_state=42)

    # Stacked Autoencoder for supervised learning
    input_dim = X_unlabelled.shape[1]
    encoding_dims = 128
    activation = 'relu'

    encoder_input = layers.Input(shape=(input_dim,))
    x = encoder_input
    for dim in range(encoding_dims + 1):
        x = layers.Dense(dim, activation=activation)(x)
    encoder_output = x
    encoder = models.Model(encoder_input, encoder_output)

    decoder_input = layers.Input(shape=(encoding_dims,))
    x = decoder_input
    for dim in range(encoding_dims, input_dim):
        x = layers.Dense(dim, activation=activation)(x)
    decoder_output = layers.Dense(input_dim, activation='sigmoid')(x)
    decoder = models.Model(decoder_input, decoder_output)

    autoencoder_input = layers.Input(shape=(input_dim,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = models.Model(autoencoder_input, decoded, name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X_unlabelled, X_unlabelled, epochs=1, batch_size=32, validation_split=0.2)
    y_pred = encoder.predict(X_test)
    y_pred = decoder.predict(y_pred)
    # Create the supervised model
    reconstruction_error = autoencoder.evaluate(X_test, X_test)
    print("Reconstruction Error on test set:", reconstruction_error)
    # test_loss, test_acc = autoencoder.evaluate(X_test, y_test)  # accuracy
    # print("Accuracy Score of Unsupervised-pretraining Neural Network is:", test_acc)
    # f = f1_score(y_test, y_pred)  # f1
    # print("F1 Score of Unsupervised-pretraining Neural Network is:", f)

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
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show(block=True)
    plt.savefig('roc_curve.png')

# ROC curves for all models
y_pred_list = []  # List to store predicted probabilities for each model
labels = []
# print(Counter(supervised_learning))
y_pred_list = [supervised_learning,
               self_50, self_75, self_95, self_99,
               co_50, co_75, co_95, co_99,
               boost_50, boost_75, boost_90, boost_99]
labels = ['Supervised Learning',
          'Self_50', 'Self_75', 'Self_95', 'Self_99',
          'Co_50', 'Co_75', 'Co_95', 'Co_99',
          'Boost_50', 'Boost_75', 'Boost_90', 'Boost_99',]
plot_roc_curve(y_test, y_pred_list, labels)

