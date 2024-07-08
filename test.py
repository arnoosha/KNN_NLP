import time
import numpy as np
import pandas as pd
from hazm import *
import pickle
from train import categories_train


normalizer = Normalizer()
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()
posTagger = POSTagger(model='pos_tagger.model')


def create_DTM_Test(total_doc_test , unique_words):
    dtm_test = []
    for doc in total_doc_test:
        v = [0 for i in range(len(unique_words))]
        for i in range(len(unique_words)):
            v[i] += doc.count(unique_words[i])
        dtm_test.append(v)
    return dtm_test




class myKNN:
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
            ans = "Sport" if votes.count("Sport") > votes.count("Politics") else "Politics"
            final_output.append(ans)

        return final_output








load_xtrain_pca = "X_train_pca.sav"
X_train_pca = pickle.load(open(load_xtrain_pca, 'rb'))
y_train = categories_train



my_knn = myKNN(k=9)
my_knn.fit(X_train_pca, y_train)



# df_test = pd.read_csv("news.csv", encoding='utf-8')
# contents_test = df_test['body'].tolist()
# categories_test = df_test['type'].tolist()

df_test = pd.read_csv("nlp_test.csv", encoding='utf-8')
contents_test = df_test['Text'].tolist()
categories_test = df_test['Category'].tolist()

#
# df_test1 = pd.read_csv("Crawl.csv", encoding='utf-8')
#
# contents_test = df_test1['Context'].tolist()
# categories_test = df_test1['Tag'].tolist()
#
# df_test2 = pd.read_csv("Crawl_Politics.csv", encoding='utf-8')
# contents_test += df_test2['Context'].tolist()
# categories_test += df_test2['Tag'].tolist()


total_documents_test = []

for content, category in zip(contents_test, categories_test):
    content = normalizer.normalize(content)
    tokenized_content = tokenizer.tokenize(content)
    tagged = posTagger.tag(tokens=tokenized_content)
    new_listof_tokenized = []

    for tok in tagged:
        if (
            tok[1] != "NUM,EZ"
            and tok[1] != "NUM"
            and tok[1] != "ADP"
            and tok[1] != "CCONJ"
            and tok[1] != "PUNCT"
            and tok[1] != "VERB"
            and tok[1] != "PUNCT"
            and tok[1] != "VERB"
            and tok[1] != "SCONJ"
            and tok[1] != "DET"
            and tok[1] != "PRON"
        ):
            lemmatized_content = lemmatizer.lemmatize(tok[0])
            new_listof_tokenized.append(lemmatized_content)

    total_documents_test.append(' '.join(new_listof_tokenized))



load_unique = 'unique.sav'
unique_load =  pickle.load(open(load_unique , 'rb'))

print("create DTM test")
start_dtm = time.time()

dtm_test = create_DTM_Test(total_documents_test, unique_load)

end_dtm = time.time()
print("dtm time :")
print(end_dtm-start_dtm)


load_pcafit = "pcafit.sav"
pcafit_load = pickle.load(open(load_pcafit, 'rb'))

# X_test_scaled = scaler.transform(dtm_test)
# X_test_pca = pcafit_load.transform(X_test_scaled)



X_test_pca = pcafit_load.transform(dtm_test)


y_pred_tests = my_knn.predict(X_test_pca)

for true_label, predicted_label in zip(categories_test, y_pred_tests):
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")




def calculate_f1_score(y_true, y_pred):
    pos = "Sport"
    true_positives = sum((a == b and b == pos) for a, b in zip(y_true, y_pred))
    false_positives = sum((a != b and b == pos) for a, b in zip(y_true, y_pred))
    false_negatives = sum((a != b and b != pos) for a, b in zip(y_true, y_pred))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score

f1_test = calculate_f1_score(categories_test, y_pred_tests)
# accuracy_test = accuracy_score(categories_test, y_pred_tests)
# f1_py_test = f1_score(label_encoder.transform(categories_test), y_pred_tests)


#print(f"Accuracy for X_test: {accuracy_test}")
print(f"F1 Score manually for X_test: {f1_test}")
#print(f"F1 Score py for X_test: {f1_py_test}")