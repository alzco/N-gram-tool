"""
Combined N-gram Text Analyzer - Streamlit Application
This application combines single-document and multi-document N-gram analysis capabilities.
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from collections import Counter
from typing import Dict, List, Tuple, Optional
import os

# Import custom modules
from modules.document_processor import extract_text_from_file, preprocess_text, tokenize_for_ngrams
from modules.ngram_analyzer import (
    analyze_document_ngrams, 
    calculate_document_similarity, 
    find_distinctive_ngrams,
    generate_similarity_heatmap,
    generate_ngram_comparison_chart
)

# Set page configuration
st.set_page_config(
    page_title="N-gram Text Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, academic look
st.markdown("""
<style>
    .main {
        padding: 2rem;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .stTextInput, .stTextArea {
        background-color: #f9f9f9;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .stButton button {
        background-color: #4361ee;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        box-shadow: 0 4px 10px rgba(67, 97, 238, 0.2);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #3a56d4;
        box-shadow: 0 6px 15px rgba(67, 97, 238, 0.3);
        transform: translateY(-2px);
    }
    h1 {
        color: #2d3748;
        font-weight: 700;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    h2, h3 {
        color: #4361ee;
        font-weight: 600;
    }
    h4, h5 {
        color: #4a5568;
        font-weight: 600;
    }
    .result-container {
        border-top: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 1rem 0;
        margin: 1.5rem 0;
    }
    .chart-container {
        border-top: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 1rem 0;
        margin: 1.5rem 0;
    }
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    .toggle-container {
        border-top: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 1rem 0;
        margin: 1rem 0;
    }
    a {
        color: #4361ee;
        text-decoration: none;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        background-color: #f0f7ff;
        display: inline-block;
        margin-top: 1rem;
        transition: all 0.3s ease;
    }
    a:hover {
        background-color: #e1e9ff;
        box-shadow: 0 2px 5px rgba(67, 97, 238, 0.2);
    }
    .file-uploader {
        border-top: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 1rem 0;
        margin: 1.5rem 0;
    }
    .similarity-matrix {
        border-top: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 1rem 0;
        margin: 1.5rem 0;
    }
    .distinctive-ngrams {
        border-top: 1px solid #e2e8f0;
        padding: 1rem 0;
        margin-top: 1.2rem;
    }
    .mode-selector {
        border-bottom: 1px solid #e2e8f0;
        padding: 1rem 0;
        margin-bottom: 2rem;
        text-align: center;
    }
    .tab-content {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("N-gram Text Analyzer")

# Mode selection
st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
st.markdown("### Select Analysis Mode")
analysis_mode = st.radio(
    "Choose analysis mode:",
    options=["Single Document Analysis", "Multi-Document Comparison"],
    index=0,
    horizontal=True,
    help="Select whether to analyze a single document or compare multiple documents."
)

if analysis_mode == "Single Document Analysis":
    st.markdown("""
    <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
    Analyze N-grams in a single document. Input text directly or upload a file.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
    Compare N-gram patterns across multiple documents. Upload multiple files to analyze similarities and differences.
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Create a main column for input and results
col1 = st.container()
col2 = st.sidebar

# Sidebar with all options
with col2:
    st.markdown("### Analysis Options")
    
    # Input parameters
    n_value = st.slider(
        "N-gram Size:",
        min_value=1,
        max_value=25,
        value=3,
        step=1,
        help="The size of N-grams to generate (e.g., 1 for unigrams, 2 for bigrams, 3 for trigrams)."
    )
    
    top_n = st.slider(
        "Number of Results:",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="The number of top N-grams to display per document."
    )
    
    # Language selection
    st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
    st.markdown("##### Language Selection")
    language = st.radio(
        "Select primary text language:",
        options=["Chinese", "English"],
        index=0,
        help="Select the primary language of your texts for optimized processing."
    )
    
    # Add instructions based on language mode
    if language == "Chinese":
        st.markdown("""
        <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
        <strong>Chinese Mode:</strong> Optimized for character-level analysis of Chinese text. 
        Automatically removes punctuation and spaces by default for better results.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
        <strong>English Mode:</strong> Defaults to word-level N-gram analysis, which is more 
        meaningful for English text. Character-level analysis is also available.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Text processing options
    st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
    st.markdown("##### Text Processing Options")
    
    # Set default values based on language
    if language == "Chinese":
        remove_punctuation = st.checkbox(
            "Remove Punctuation",
            value=True,  # Default to True for Chinese
            help="Toggle to remove punctuation marks from the text before analysis."
        )
        
        remove_spaces = st.checkbox(
            "Remove Spaces",
            value=True,  # Default to True for Chinese
            help="Toggle to remove all spaces from the text before analysis."
        )
        
        remove_english = st.checkbox(
            "Remove English Letters",
            value=False,
            help="Toggle to remove all English letters (a-z, A-Z) from the text before analysis."
        )
    else:  # English
        remove_punctuation = st.checkbox(
            "Remove Punctuation",
            value=False,
            help="Toggle to remove punctuation marks from the text before analysis."
        )
        
        remove_spaces = st.checkbox(
            "Remove Spaces",
            value=False,
            help="Toggle to remove all spaces from the text before analysis (useful for character-level N-grams)."
        )
        
        word_level = st.checkbox(
            "Word-level N-grams",
            value=True,  # Default to True for English
            help="Toggle to analyze word-level N-grams instead of character-level N-grams."
        )
    
    remove_numbers = st.checkbox(
        "Remove Arabic Numerals",
        value=False,
        help="Toggle to remove all Arabic numerals (0-9) from the text before analysis."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Color theme selection
    st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
    st.markdown("##### Visualization Options")
    
    color_theme = st.selectbox(
        "Color Theme:",
        options=[
            "Modern Blue", "Fresh Mint", "Sunset", "Berry", 
            "Pastel", "Dark Mode", "Vibrant", "Classic"
        ],
        index=0,
        help="Select a color theme for the visualization."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Multi-document specific options
    if analysis_mode == "Multi-Document Comparison":
        st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
        st.markdown("##### Advanced Analysis Options")
        
        similarity_method = st.selectbox(
            "Similarity Calculation Method:",
            options=["Cosine Similarity", "Jaccard Similarity", "Overlap Coefficient"],
            index=0,
            help="Method used to calculate similarity between documents."
        )
        
        min_distinctive_freq = st.slider(
            "Minimum Frequency for Distinctive N-grams:",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Minimum frequency for an N-gram to be considered as distinctive to a document."
        )
        
        # Heatmap color selection
        heatmap_color = st.selectbox(
            "Similarity Heatmap Color Scheme:",
            options=[
                "YlGnBu", "viridis", "plasma", "inferno", "magma", 
                "cividis", "mako", "rocket", "Blues", "Greens"
            ],
            index=0,
            help="Color scheme for the similarity heatmap."
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Main area content based on selected mode
with col1:
    if analysis_mode == "Single Document Analysis":
        # Import single document analysis function
        from single_document_mode import run_single_document_mode
        
        # Run single document mode
        run_single_document_mode(
            language=language,
            n_value=n_value,
            top_n=top_n,
            remove_punctuation=remove_punctuation,
            remove_spaces=remove_spaces,
            remove_english=remove_english if language == "Chinese" else None,
            word_level=word_level if language == "English" else None,
            remove_numbers=remove_numbers,
            color_theme=color_theme
        )
    else:  # Multi-Document Comparison
        # Import multi-document analysis function
        from multi_document_mode import run_multi_document_mode
        
        # Run multi-document mode
        run_multi_document_mode(
            language=language,
            n_value=n_value,
            top_n=top_n,
            remove_punctuation=remove_punctuation,
            remove_spaces=remove_spaces,
            remove_english=remove_english if language == "Chinese" else None,
            word_level=word_level if language == "English" else None,
            remove_numbers=remove_numbers,
            color_theme=color_theme,
            similarity_method=similarity_method,
            min_distinctive_freq=min_distinctive_freq,
            heatmap_color=heatmap_color
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; padding: 1rem 0;">
    <p>N-gram Text Analyzer | Developed for Academic Research</p>
</div>
""", unsafe_allow_html=True)
