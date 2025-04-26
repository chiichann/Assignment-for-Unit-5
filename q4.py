import wikipedia
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

topics = [
    "Data Science",
    "Data Visualization",
    "Quantum computing",
    "Big data",
    "Augmented reality"
]

def get_doc_vector(doc_tokens):
    vectors = [w2v_model.wv[word] for word in doc_tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

documents = [wikipedia.summary(topic) for topic in topics]
tokenized_documents = {i: re.findall(r'\b\w+\b', doc.lower()) for i, doc in enumerate(documents)}

w2v_model = Word2Vec(sentences=tokenized_documents.values(), vector_size=100, window=5, min_count=1, workers=4, sg=1)
w2v_model.save("word2vec.model")

X = np.array([get_doc_vector(tokens) for tokens in tokenized_documents.values()])

y_labels = ['data', 'data', 'tech', 'data', 'visual']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, labels=np.unique(y)))
