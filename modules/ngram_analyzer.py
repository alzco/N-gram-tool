"""
Advanced N-gram Analyzer Module
This module provides functions for advanced N-gram analysis across multiple documents.
"""
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
# 添加回必要的导入
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    """
    Generate n-grams from a list of tokens.
    
    Args:
        tokens: List of tokens (characters or words)
        n: Size of n-grams
        
    Returns:
        List[str]: List of n-grams
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        if isinstance(tokens[i], str):
            ngram = ''.join(tokens[i:i+n]) if all(isinstance(t, str) and len(t) == 1 for t in tokens[i:i+n]) else ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
    return ngrams

@st.cache_data
def analyze_document_ngrams(tokens: List[str], n: int, top_n: int) -> List[Tuple[str, int]]:
    """
    Analyze n-grams in a document.
    
    Args:
        tokens: List of tokens (characters or words)
        n: Size of n-grams
        top_n: Number of top n-grams to return
        
    Returns:
        List[Tuple[str, int]]: List of (n-gram, frequency) tuples
    """
    # Generate n-grams
    ngrams = generate_ngrams(tokens, n)
    
    # Count frequencies
    ngram_counts = Counter(ngrams)
    
    # Get top n-grams
    top_ngrams = ngram_counts.most_common(top_n)
    
    return top_ngrams

@st.cache_data
def calculate_document_similarity(doc_ngrams: Dict[str, Counter], method: str = 'cosine') -> pd.DataFrame:
    """
    Calculate similarity between documents based on their n-gram profiles.
    
    Args:
        doc_ngrams: Dictionary mapping document names to their n-gram Counter objects
        method: Similarity method ('cosine', 'jaccard', or 'overlap')
        
    Returns:
        pd.DataFrame: Similarity matrix
    """
    doc_names = list(doc_ngrams.keys())
    n_docs = len(doc_names)
    similarity_matrix = np.zeros((n_docs, n_docs))
    
    # Create a set of all n-grams across all documents
    all_ngrams = set()
    for counter in doc_ngrams.values():
        all_ngrams.update(counter.keys())
    
    # Convert to a sorted list for consistent indexing
    all_ngrams_list = sorted(list(all_ngrams))
    
    # Create vectors for each document
    doc_vectors = {}
    for doc_name, counter in doc_ngrams.items():
        vector = [counter.get(ngram, 0) for ngram in all_ngrams_list]
        doc_vectors[doc_name] = vector
    
    # Calculate similarity
    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity
            else:
                vec1 = doc_vectors[doc_names[i]]
                vec2 = doc_vectors[doc_names[j]]
                
                if method == 'cosine':
                    # Cosine similarity
                    dot_product = sum(a * b for a, b in zip(vec1, vec2))
                    norm1 = sum(a * a for a in vec1) ** 0.5
                    norm2 = sum(b * b for b in vec2) ** 0.5
                    similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
                    
                elif method == 'jaccard':
                    # Jaccard similarity
                    nonzero1 = set(i for i, v in enumerate(vec1) if v > 0)
                    nonzero2 = set(i for i, v in enumerate(vec2) if v > 0)
                    intersection = len(nonzero1.intersection(nonzero2))
                    union = len(nonzero1.union(nonzero2))
                    similarity = intersection / union if union > 0 else 0
                    
                elif method == 'overlap':
                    # Overlap coefficient
                    nonzero1 = set(i for i, v in enumerate(vec1) if v > 0)
                    nonzero2 = set(i for i, v in enumerate(vec2) if v > 0)
                    intersection = len(nonzero1.intersection(nonzero2))
                    min_size = min(len(nonzero1), len(nonzero2))
                    similarity = intersection / min_size if min_size > 0 else 0
                
                similarity_matrix[i, j] = similarity
    
    return pd.DataFrame(similarity_matrix, index=doc_names, columns=doc_names)

def find_distinctive_ngrams(doc_ngrams: Dict[str, Counter], min_freq: int = 2) -> Dict[str, List[Tuple[str, int]]]:
    """
    Find distinctive n-grams that appear in one document but not in others.
    
    Args:
        doc_ngrams: Dictionary mapping document names to their n-gram Counter objects
        min_freq: Minimum frequency for an n-gram to be considered
        
    Returns:
        Dict[str, List[Tuple[str, int]]]: Dictionary mapping document names to their distinctive n-grams
    """
    distinctive_ngrams = {}
    
    for doc_name, counter in doc_ngrams.items():
        # Get all other documents' n-grams
        other_docs_ngrams = set()
        for other_doc, other_counter in doc_ngrams.items():
            if other_doc != doc_name:
                other_docs_ngrams.update(other_counter.keys())
        
        # Find n-grams unique to this document with frequency >= min_freq
        unique_ngrams = [(ngram, freq) for ngram, freq in counter.items() 
                         if ngram not in other_docs_ngrams and freq >= min_freq]
        
        # Sort by frequency
        unique_ngrams.sort(key=lambda x: x[1], reverse=True)
        
        distinctive_ngrams[doc_name] = unique_ngrams
    
    return distinctive_ngrams

def generate_similarity_heatmap(similarity_matrix: pd.DataFrame) -> BytesIO:
    """
    Generate a heatmap visualization of document similarity.
    
    Args:
        similarity_matrix: DataFrame containing similarity scores
        
    Returns:
        BytesIO: Image buffer containing the heatmap
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1, 
                linewidths=.5, fmt=".2f", cbar_kws={"shrink": .8})
    plt.title("Document Similarity Matrix", fontsize=16)
    plt.tight_layout()
    
    # Save plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()
    
    return buf

def generate_ngram_comparison_chart(combined_df: pd.DataFrame, n_value: int) -> BytesIO:
    """
    Generate a bar chart comparing n-gram frequencies across documents.
    
    Args:
        combined_df: DataFrame with columns 'N-gram', 'Frequency', 'Document'
        n_value: Size of n-grams
        
    Returns:
        BytesIO: Image buffer containing the chart
    """
    plt.figure(figsize=(12, 8))
    
    # Pivot the data for easier plotting
    pivot_df = combined_df.pivot(index='N-gram', columns='Document', values='Frequency')
    pivot_df = pivot_df.fillna(0)
    
    # Sort by total frequency across all documents
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('Total', ascending=False)
    pivot_df = pivot_df.drop('Total', axis=1)
    
    # Take top 15 for better readability
    pivot_df = pivot_df.head(15)
    
    # Plot
    ax = pivot_df.plot(kind='barh', figsize=(12, 8), width=0.8)
    plt.title(f"Top {n_value}-grams Across Documents", fontsize=16)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("N-gram", fontsize=12)
    plt.legend(title="Document", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()
    
    return buf
