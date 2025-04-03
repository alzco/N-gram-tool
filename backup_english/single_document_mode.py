"""
Single Document Analysis Mode
This module provides the functionality for analyzing N-grams in a single document.
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from collections import Counter
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO

from modules.document_processor import extract_text_from_file, preprocess_text, tokenize_for_ngrams

def generate_ngrams(tokens, n):
    """Generate n-grams from a list of tokens."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        if isinstance(tokens[i], str):
            ngram = ''.join(tokens[i:i+n]) if all(isinstance(t, str) and len(t) == 1 for t in tokens[i:i+n]) else ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
    return ngrams

def analyze_text(text, n, top_n, language="Chinese", remove_punctuation=True, 
                remove_spaces=True, remove_english=False, word_level=False, 
                remove_numbers=False):
    """Analyze text to find the most frequent n-grams."""
    # Preprocess text
    processed_text = preprocess_text(
        text, 
        language=language,
        remove_punctuation=remove_punctuation,
        remove_spaces=remove_spaces,
        remove_english=remove_english if language == "Chinese" else False,
        remove_numbers=remove_numbers
    )
    
    # Tokenize text
    tokens = tokenize_for_ngrams(processed_text, language, word_level if language == "English" else False)
    
    # Generate n-grams
    ngrams = generate_ngrams(tokens, n)
    
    # Count frequencies
    ngram_counts = Counter(ngrams)
    
    # Get top n-grams
    top_ngrams = ngram_counts.most_common(top_n)
    
    return top_ngrams

def run_single_document_mode(language, n_value, top_n, remove_punctuation, remove_spaces, 
                            remove_english, word_level, remove_numbers, color_theme):
    """Run the single document analysis mode."""
    
    st.markdown("## Single Document N-gram Analysis")
    
    # Input options
    input_method = st.radio(
        "Select input method:",
        options=["Text Input", "File Upload"],
        index=0,
        horizontal=True,
        help="Choose whether to input text directly or upload a file."
    )
    
    # Text input area
    if input_method == "Text Input":
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        if language == "Chinese":
            text_placeholder = "请在此输入中文文本进行分析..."
        else:
            text_placeholder = "Enter English text here for analysis..."
            
        text_input = st.text_area(
            "",  # 空标签，不显示额外的标题
            height=200,
            placeholder=text_placeholder
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze button
        analyze_button = st.button("Analyze Text", type="primary")
        
        # Process input and display results
        if analyze_button and text_input:
            with st.spinner("Analyzing text..."):
                # Process text based on user preferences
                results = analyze_text(
                    text_input,
                    n_value,
                    top_n,
                    language=language,
                    remove_punctuation=remove_punctuation,
                    remove_spaces=remove_spaces,
                    remove_english=remove_english if language == "Chinese" else False,
                    word_level=word_level if language == "English" else False,
                    remove_numbers=remove_numbers
                )
                
                # Display results
                display_results(results, n_value, color_theme)
        elif analyze_button and not text_input:
            st.warning("Please enter some text to analyze.")
    
    # File upload
    else:
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        st.markdown("### Upload a Document")
        st.markdown("Upload a document to analyze its N-gram patterns. Supported formats: .txt, .md, .docx, .pdf")
        
        uploaded_file = st.file_uploader(
            "Upload a document", 
            type=["txt", "md", "docx", "pdf"]
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process and display results
        if uploaded_file:
            # Show uploaded file name
            st.markdown(f"**Uploaded file:** {uploaded_file.name}")
            
            # Analyze button
            analyze_button = st.button("Analyze Document", type="primary")
            
            if analyze_button:
                with st.spinner("Analyzing document..."):
                    # Extract text from file
                    text = extract_text_from_file(uploaded_file, uploaded_file.name)
                    
                    # Process text based on user preferences
                    results = analyze_text(
                        text,
                        n_value,
                        top_n,
                        language=language,
                        remove_punctuation=remove_punctuation,
                        remove_spaces=remove_spaces,
                        remove_english=remove_english if language == "Chinese" else False,
                        word_level=word_level if language == "English" else False,
                        remove_numbers=remove_numbers
                    )
                    
                    # Display results
                    display_results(results, n_value, color_theme)
        else:
            st.info("Please upload a document to begin analysis.")

def display_results(results, n_value, color_theme):
    """Display the analysis results."""
    
    if not results:
        st.warning("No results found. Try adjusting your parameters or using a different text.")
        return
    
    st.markdown("## Analysis Results")
    
    # Create a dataframe for the results
    df = pd.DataFrame(results, columns=["N-gram", "Frequency"])
    
    # Display the dataframe
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown(f"### Top {len(results)} {n_value}-grams")
    
    # Convert the maximum frequency to a Python int to avoid JSON serialization issues
    max_freq = int(df["Frequency"].max())
    
    # Style the dataframe
    styled_df = df.copy()
    
    # Display the dataframe without progress bars
    st.dataframe(
        styled_df,
        use_container_width=True,
        column_config={
            "N-gram": st.column_config.TextColumn("N-gram Term"),
            "Frequency": st.column_config.NumberColumn(
                "Frequency",
                format="%d",
            ),
        },
        hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create a visualization
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### Frequency Visualization")
    
    # Set color theme
    if color_theme == "Modern Blue":
        colors = px.colors.sequential.Blues
    elif color_theme == "Fresh Mint":
        colors = px.colors.sequential.Mint
    elif color_theme == "Sunset":
        colors = px.colors.sequential.Sunset
    elif color_theme == "Berry":
        colors = px.colors.sequential.Purp
    elif color_theme == "Pastel":
        colors = px.colors.sequential.PuBu
    elif color_theme == "Dark Mode":
        colors = px.colors.sequential.gray
    elif color_theme == "Vibrant":
        colors = px.colors.sequential.Viridis
    else:  # Classic
        colors = px.colors.sequential.Blues_r
    
    # Create a bar chart
    fig = px.bar(
        df,
        x="N-gram",
        y="Frequency",
        color="Frequency",
        color_continuous_scale=colors,
        labels={"N-gram": "N-gram Term", "Frequency": "Frequency"},
        height=500,
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Frequency",
        coloraxis_showscale=False,  # Remove the colorbar
        hovermode="closest",
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='white',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add download links for the results
    st.markdown("### Export Results")
    
    # CSV download
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name=f"ngram_analysis_{n_value}gram_results.csv",
        mime="text/csv",
    )
