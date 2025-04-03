"""
Multi-Document Analysis Mode
This module provides the functionality for comparing N-grams across multiple documents.
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

from modules.document_processor import extract_text_from_file, preprocess_text, tokenize_for_ngrams
from modules.ngram_analyzer import (
    analyze_document_ngrams, 
    calculate_document_similarity, 
    find_distinctive_ngrams,
    generate_similarity_heatmap,
    generate_ngram_comparison_chart
)

def run_multi_document_mode(language, n_value, top_n, remove_punctuation, remove_spaces, 
                           remove_english, word_level, remove_numbers, color_theme,
                           similarity_method, min_distinctive_freq, heatmap_color):
    """Run the multi-document comparison mode."""
    
    st.markdown("## Multi-Document N-gram Comparison")
    
    # Document upload area
    st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
    st.markdown("### Upload Documents")
    st.markdown("Upload multiple documents to compare their N-gram profiles. Supported formats: .txt, .md, .docx, .pdf")
    
    uploaded_files = st.file_uploader(
        "Upload documents for comparison", 
        accept_multiple_files=True,
        type=["txt", "md", "docx", "pdf"]
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process and display results
    if uploaded_files:
        # Show number of uploaded documents
        st.markdown(f"**{len(uploaded_files)} documents uploaded**")
        
        # Display file names
        file_names = [file.name for file in uploaded_files]
        st.markdown("#### Uploaded Documents:")
        for i, name in enumerate(file_names, 1):
            st.markdown(f"{i}. {name}")
        
        # Analyze button
        analyze_button = st.button("Analyze Documents", type="primary")
        
        if analyze_button:
            with st.spinner("Analyzing documents..."):
                # Process each document
                documents = {}
                for file in uploaded_files:
                    # Extract text from file
                    text = extract_text_from_file(file, file.name)
                    
                    # Store document content
                    documents[file.name] = text
                
                # Process and analyze each document
                processed_docs = {}
                doc_tokens = {}
                doc_ngrams = {}
                doc_results = {}
                
                for doc_name, text in documents.items():
                    # Preprocess text based on user preferences
                    if language == "Chinese":
                        processed_text = preprocess_text(
                            text, 
                            language=language,
                            remove_punctuation=remove_punctuation,
                            remove_spaces=remove_spaces,
                            remove_english=remove_english,
                            remove_numbers=remove_numbers
                        )
                        word_level_param = False  # Character-level for Chinese
                    else:  # English
                        processed_text = preprocess_text(
                            text, 
                            language=language,
                            remove_punctuation=remove_punctuation,
                            remove_spaces=remove_spaces,
                            remove_numbers=remove_numbers
                        )
                        word_level_param = word_level  # User selection for English
                    
                    # Store processed text
                    processed_docs[doc_name] = processed_text
                    
                    # Tokenize text
                    tokens = tokenize_for_ngrams(processed_text, language, word_level_param)
                    doc_tokens[doc_name] = tokens
                    
                    # Analyze n-grams
                    results = analyze_document_ngrams(tokens, n_value, top_n)
                    doc_results[doc_name] = results
                    
                    # Store n-gram counters for similarity calculation
                    ngrams = [ngram for ngram, _ in results]
                    freqs = [freq for _, freq in results]
                    doc_ngrams[doc_name] = {ngram: freq for ngram, freq in zip(ngrams, freqs)}
                
                # Create combined dataframe for comparison
                combined_df = pd.DataFrame()
                for doc_name, results in doc_results.items():
                    df = pd.DataFrame(results, columns=["N-gram", "Frequency"])
                    df["Document"] = doc_name
                    combined_df = pd.concat([combined_df, df])
                
                # Calculate document similarity
                similarity_method_map = {
                    "Cosine Similarity": "cosine",
                    "Jaccard Similarity": "jaccard",
                    "Overlap Coefficient": "overlap"
                }
                method = similarity_method_map[similarity_method]
                
                # Convert doc_ngrams to Counter objects
                doc_counters = {doc: Counter(ngrams) for doc, ngrams in doc_ngrams.items()}
                
                similarity_matrix = calculate_document_similarity(doc_counters, method)
                
                # Find distinctive n-grams
                distinctive_ngrams = find_distinctive_ngrams(doc_counters, min_distinctive_freq)
                
                # Display results
                st.markdown("## Analysis Results")
                
                # 1. N-gram Comparison Visualization
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("### N-gram Frequency Comparison")
                
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
                
                # Create a comparison chart
                # Pivot the data for easier plotting
                pivot_df = combined_df.pivot_table(
                    index="N-gram", 
                    columns="Document", 
                    values="Frequency", 
                    fill_value=0
                )
                
                # Sort by total frequency across all documents
                pivot_df['Total'] = pivot_df.sum(axis=1)
                pivot_df = pivot_df.sort_values('Total', ascending=False)
                pivot_df = pivot_df.drop('Total', axis=1)
                
                # Take top 15 for better readability
                pivot_df = pivot_df.head(15)
                
                # Create a horizontal bar chart
                fig = go.Figure()
                
                # Add traces for each document
                for doc_name in pivot_df.columns:
                    fig.add_trace(go.Bar(
                        y=pivot_df.index,
                        x=pivot_df[doc_name],
                        name=doc_name,
                        orientation='h',
                        hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>',
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"Top {n_value}-grams Across Documents",
                    xaxis_title="Frequency",
                    yaxis_title="N-gram",
                    barmode='group',
                    height=600,
                    legend_title="Document",
                    hovermode="closest",
                    margin=dict(l=40, r=40, t=40, b=40),
                    plot_bgcolor='white',
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate and save the comparison chart
                comparison_chart_buf = generate_ngram_comparison_chart(combined_df, n_value)
                
                # Add a download link for the chart
                st.markdown("#### Download Comparison Chart")
                st.markdown("Click the button below to download the comparison chart as a PNG image:")
                st.download_button(
                    label="Download Comparison Chart",
                    data=comparison_chart_buf,
                    file_name=f"ngram_comparison_{n_value}gram.png",
                    mime="image/png",
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 2. Document Similarity Matrix
                st.markdown('<div class="similarity-matrix">', unsafe_allow_html=True)
                st.markdown("### Document Similarity Matrix")
                st.markdown(f"Using {similarity_method} to compare document N-gram profiles:")
                
                # Display similarity matrix as a heatmap
                fig_heatmap = px.imshow(
                    similarity_matrix,
                    text_auto=".2f",
                    color_continuous_scale=heatmap_color,
                    labels=dict(x="Document", y="Document", color="Similarity"),
                    zmin=0, zmax=1
                )
                
                fig_heatmap.update_layout(
                    height=500,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Generate and save the similarity heatmap
                similarity_heatmap_buf = generate_similarity_heatmap(similarity_matrix)
                
                # Add a download link for the heatmap
                st.markdown("#### Download Similarity Heatmap")
                st.markdown("Click the button below to download the similarity heatmap as a PNG image:")
                st.download_button(
                    label="Download Similarity Heatmap",
                    data=similarity_heatmap_buf,
                    file_name=f"similarity_heatmap_{n_value}gram.png",
                    mime="image/png",
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 3. Distinctive N-grams
                st.markdown("### Distinctive N-grams")
                st.markdown(f"N-grams that appear uniquely in each document (minimum frequency: {min_distinctive_freq}):")
                
                for doc_name, unique_ngrams in distinctive_ngrams.items():
                    with st.expander(f"Distinctive N-grams in {doc_name}"):
                        if unique_ngrams:
                            # Create a dataframe for the distinctive n-grams
                            unique_df = pd.DataFrame(unique_ngrams, columns=["N-gram", "Frequency"])
                            
                            # Display the dataframe
                            st.dataframe(
                                unique_df,
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
                            
                            # Create a bar chart for the distinctive n-grams
                            if len(unique_ngrams) > 0:
                                fig_unique = px.bar(
                                    unique_df.head(15),  # Show top 15 for readability
                                    x="N-gram",
                                    y="Frequency",
                                    color="Frequency",
                                    color_continuous_scale=colors,
                                    labels={"N-gram": "N-gram Term", "Frequency": "Frequency"},
                                    height=400,
                                )
                                
                                fig_unique.update_layout(
                                    xaxis_title="",
                                    yaxis_title="Frequency",
                                    coloraxis_showscale=False,  # Remove the colorbar
                                    hovermode="closest",
                                    margin=dict(l=40, r=40, t=40, b=40),
                                    plot_bgcolor='white',
                                )
                                
                                st.plotly_chart(fig_unique, use_container_width=True)
                        else:
                            st.markdown("No distinctive N-grams found for this document.")
                
                # 4. Individual Document Results
                st.markdown("### Individual Document Results")
                
                for doc_name, results in doc_results.items():
                    with st.expander(f"Results for {doc_name}"):
                        # Create a dataframe for the results
                        doc_df = pd.DataFrame(results, columns=["N-gram", "Frequency"])
                        
                        # Display the dataframe
                        st.dataframe(
                            doc_df,
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
                        
                        # Create a bar chart for the document
                        fig_doc = px.bar(
                            doc_df.head(15),  # Show top 15 for readability
                            x="N-gram",
                            y="Frequency",
                            color="Frequency",
                            color_continuous_scale=colors,
                            labels={"N-gram": "N-gram Term", "Frequency": "Frequency"},
                            height=400,
                        )
                        
                        fig_doc.update_layout(
                            xaxis_title="",
                            yaxis_title="Frequency",
                            coloraxis_showscale=False,  # Remove the colorbar
                            hovermode="closest",
                            margin=dict(l=40, r=40, t=40, b=40),
                            plot_bgcolor='white',
                        )
                        
                        st.plotly_chart(fig_doc, use_container_width=True)
                        
                        # Add download links for the results
                        st.markdown("#### Download Results")
                        
                        # CSV download
                        csv = doc_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{doc_name}_{n_value}gram_results.csv",
                            mime="text/csv",
                        )
                
                # 5. Export All Results
                st.markdown("### Export All Results")
                
                # Prepare combined CSV
                all_results_csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="Download All Results (CSV)",
                    data=all_results_csv,
                    file_name=f"all_documents_{n_value}gram_results.csv",
                    mime="text/csv",
                )
                
                # Prepare similarity matrix CSV
                similarity_csv = similarity_matrix.to_csv(index=True)
                st.download_button(
                    label="Download Similarity Matrix (CSV)",
                    data=similarity_csv,
                    file_name=f"document_similarity_{n_value}gram.csv",
                    mime="text/csv",
                )
    else:
        st.info("Please upload documents to begin analysis.")
