import sys
import numpy as np
import pandas as pd
from hazm import *
from sklearn.decomposition import PCA
import pickle


sys.stdout.reconfigure(encoding="utf-8")

normalizer = Normalizer()
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()
posTagger = POSTagger(model='pos_tagger.model')


df_train = pd.read_csv("nlp_train.csv", encoding='utf-8')
df_train = df_train[5000:15000]
contents_train = df_train['Text'].tolist()
categories_train = df_train['Category'].tolist()


total_documents_train = []


for content, category in zip(contents_train, categories_train):
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

    total_documents_train.append(' '.join(new_listof_tokenized))

def create_DTM(documents, word_index_mapping=None):
    unique_words = []
    dtm = []

    if word_index_mapping is None:
        word_index_mapping = {}

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

    return dense_matrix, unique_words, word_index_mapping





dtm_train, unique_words_train , word_index_mapping_train = create_DTM([doc for doc in total_documents_train])

save_unique_words = 'unique.sav'
pickle.dump(unique_words_train, open(save_unique_words, 'wb'))


X_train = dtm_train
#X_train, X_test, y_train, y_test = train_test_split(dtm_train, categories_train, test_size=0.3, random_state=40)




pca = PCA(n_components=1000)
pcafit = pca.fit(X_train)

save_pcafit = "pcafit.sav"

pickle.dump(pcafit, open(save_pcafit, 'wb'))

X_train_pca = pca.transform(X_train)




save_X_train = "X_train_pca.sav"
pickle.dump(X_train_pca, open(save_X_train, 'wb'))