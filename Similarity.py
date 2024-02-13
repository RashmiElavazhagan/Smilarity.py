# -------------------------------------------------------------------------
# AUTHOR: Rashmi Elavazhagan            
# FILENAME: similarity
# SPECIFICATION: Performs term frequency and calculates cosine similarity
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 60 Minutes
# -----------------------------------------------------------*/

# Importing some Python libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"

# Use the following words as terms to create your document-term matrix
terms = ['soccer', 'favorite', 'sport', 'like', 'one', 'support', 'olympic', 'games']

# Create a CountVectorizer object to convert text documents to a matrix of term frequencies
vectorizer = CountVectorizer(vocabulary=terms)
doc_term_matrix = vectorizer.fit_transform([doc1, doc2, doc3, doc4])

# Calculate pairwise cosine similarities
cos_similarities = cosine_similarity(doc_term_matrix)

# Find the indices of the highest cosine similarity (excluding comparisons of a document with itself)
np.fill_diagonal(cos_similarities, -1)  # Set diagonal elements to -1 to exclude comparisons of a document with itself
most_similar_indices = np.unravel_index(np.argmax(cos_similarities), cos_similarities.shape)
doc1_index, doc2_index = most_similar_indices

# Print the most similar documents and their cosine similarity
print("The most similar documents are: doc{} and doc{} with cosine similarity = {:.2f}".format(doc1_index + 1, doc2_index + 1, cos_similarities[doc1_index, doc2_index]))