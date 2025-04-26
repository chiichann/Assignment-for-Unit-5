import wikipedia
import re
from collections import Counter
from math import log, sqrt

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

def cosine_similarity(vec1, vec2, vocab):
    dot_product = sum(vec1[term] * vec2[term] for term in vocab)
    vec1_len = sqrt(sum(vec1[term]**2 for term in vocab))
    vec2_len = sqrt(sum(vec2[term]**2 for term in vocab))
    if vec1_len == 0 or vec2_len == 0:
        return 0.0
    return dot_product / (vec1_len * vec2_len)

tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]
idf = compute_idf(tokenized_docs, vocabulary)
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]

most_similar_score = -1
most_similar_pair = (None, None)

for i in range(len(tfidf_vectors)):
    for j in range(i + 1, len(tfidf_vectors)):
        sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j], vocabulary)
        print(f"Document {i+1} vs Document {j+1}: {sim:.4f}")
        if sim > most_similar_score:
            most_similar_score = sim
            most_similar_pair = (i, j)

print(f"\nMost similar documents: {topics[most_similar_pair[0]]} and {topics[most_similar_pair[1]]} with similarity score of {most_similar_score:.4f}")
