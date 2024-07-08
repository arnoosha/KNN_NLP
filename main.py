import sys

import numpy as np
import pandas as pd
from hazm import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import pickle
from sklearn.preprocessing import LabelEncoder


sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("nlp_train.csv", encoding='utf-8')
df = df[9000:11000]
contents = df['Text'].tolist()
categories = df['Category'].tolist()

total_documents = []

normalizer = Normalizer()
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()
posTagger = POSTagger(model='pos_tagger.model')
for content, category in zip(contents, categories):
    content = normalizer.normalize(content)
    tokenized_content = tokenizer.tokenize(content)
    tagged = posTagger.tag(tokens=tokenized_content)
    new_listof_tokenized = []

    for tok in tagged:
        if (tok[1] != "NUM,EZ" and tok[1] != "NUM" and tok[1] != "ADP" and tok[1] != "CCONJ" and tok[1] != "PUNCT" and
                tok[1] != "VERB" and tok[1] != "PUNCT" and tok[1] != "VERB" and tok[1] != "SCONJ" and tok[1] != "DET" and
                tok[1] != "PRON"):
            lemmatized_content = lemmatizer.lemmatize(tok[0])
            new_listof_tokenized.append(lemmatized_content)

    total_documents.append((' '.join(new_listof_tokenized), category))





def create_DTM(documents):
    unique_words = []
    word_index_mapping = {}
    dtm = []

    for doc in documents:
        doc_counts = {}
        for word in doc.split():
            if word not in word_index_mapping:
                word_index_mapping[word] = len(unique_words)
                unique_words.append(word)

            word_index = word_index_mapping[word]
            doc_counts[word_index] = doc_counts.get(word_index, 0) + 1

        dtm.append(doc_counts)

    dense_matrix = np.zeros((len(dtm), len(unique_words)), dtype=int)
    for i, doc_counts in enumerate(dtm):
        for word_index, count in doc_counts.items():
            dense_matrix[i, word_index] = count

    return dense_matrix , [str(word_index) for word_index in range(len(unique_words))]




dtm , unique_words = create_DTM([doc for doc, _ in total_documents])




y = [category for _, category in total_documents]


X_train, X_test, y_train, y_test = train_test_split(dtm, y, test_size=0.3, random_state=40)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=1000)
pcafit = pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Save unique words
save_unique_words = 'unique.sav'
pickle.dump(unique_words, open(save_unique_words, 'wb'))

# Save pcafit
save_pcafit = "pcafit.sav"
pickle.dump(pcafit, open(save_pcafit, 'wb'))

# Save X_train
save_X_train = "X_train_pca.sav"
pickle.dump(X_train_pca, open(save_X_train, 'wb'))



class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, X1, X2):
        return np.sqrt(np.sum((X1 - X2) ** 2))

    def predict(self, X_test):
        final_output = []
        for i in range(len(X_test)):
            d = []
            votes = []
            for j in range(len(self.X_train)):
                dist = self.distance(self.X_train[j], X_test[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.k]
            for _, j in d:
                votes.append(self.y_train[j])
            ans = np.bincount(votes).argmax()
            final_output.append(ans)

        return final_output

    def score(self, X_test, y_test):
        predictions = np.array(self.predict(X_test))
        return np.mean(predictions == y_test)


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

knn = KNN(k=3)
knn.fit(X_train_pca, y_train_encoded)
y_pred_encoded = knn.predict(X_test_pca)

y_pred = label_encoder.inverse_transform(y_pred_encoded)



def f1(actual, predicted, label):


    tp = np.sum((actual == label) & (predicted == label))
    fp = np.sum((actual != label) & (predicted == label))
    fn = np.sum((actual==label) & (predicted != label))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1


def f1_macro(actual, predicted):
    return np.mean([f1(actual, predicted, label)
        for label in np.unique(actual)])

def calculate_f1_score(y_true, y_pred):
    true_positives = sum((a == 1 and b == 1) for a, b in zip(y_true, y_pred))
    false_positives = sum((a == 0 and b == 1) for a, b in zip(y_true, y_pred))
    false_negatives = sum((a == 1 and b == 0) for a, b in zip(y_true, y_pred))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score

for true_label, predicted_label in zip(y_test, y_pred):
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")

# Evaluate accuracy and F1 score
f1 = f1_macro(y_test_encoded, y_pred_encoded)
accuracy = accuracy_score(y_test, y_pred)
f1_py = f1_score(y_test_encoded, y_pred_encoded)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score manually: {f1}")
print(f"F1 Score py: {f1_py}")
print("Confusion Matrix:")
print(cm)
