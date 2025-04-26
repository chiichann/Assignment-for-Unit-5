import wikipedia
import re
from collections import Counter
from math import log

topics = [
    "Data Science",
    "Data Visualization",
    "Quantum computing",
    "Big data",
    "Augmented reality"
]

documents = []
for topic in topics:
    try:
        summary = wikipedia.summary(topic)
        documents.append(summary)
    except Exception as e:
        print(f"Error fetching summary for {topic}: {e}")

tokenized_docs = [re.findall(r'\b\w+\b', doc.lower()) for doc in documents]
vocabulary = set(word for doc in tokenized_docs for word in doc)

def compute_tf(tokens, vocab):
    count = Counter(tokens)
    total_terms = len(tokens)
    return { term: count[term] / total_terms for term in vocab }

def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        df = sum(term in doc for doc in tokenized_docs)
        idf_dict[term] = log(N / (df or 1))
    return idf_dict

def compute_tfidf(tf_vector, idf, vocab):
    return { term: tf_vector[term] * idf[term] for term in vocab }

tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]
idf = compute_idf(tokenized_docs, vocabulary)
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]

for i, doc in enumerate(documents):
    print(f"\nDocument {i+1} ({topics[i]}):\n{doc}")

# a. Term-document matrix using raw frequency
for i, tf_vector in enumerate(tf_vectors):
    print(f"\nDocument {i+1}:")
    for term, freq in tf_vector.items():
        if freq > 0:
            print(f"{term}: {freq}")

# b. Term-document matrix using TF-IDF weights
for i, tfidf_vector in enumerate(tfidf_vectors):
    print(f"\nDocument {i+1}:")
    for term, weight in tfidf_vector.items():
        if weight > 0:
            print(f"{term}: {weight}")

# Print Inverse Document Frequency
for term, idf_value in idf.items():
    print(f"{term}: {idf_value}")
