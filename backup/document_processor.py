"""
Document Processor Module
This module provides functions to extract text from various document formats.
"""
import docx2txt
import PyPDF2
import re
import os
from typing import Dict, List, Tuple, Optional

def extract_text_from_file(file, file_name: str) -> str:
    """
    Extract text from various file formats.
    
    Args:
        file: File object from Streamlit file uploader
        file_name: Name of the file
        
    Returns:
        str: Extracted text content
    """
    # Get file extension
    _, ext = os.path.splitext(file_name.lower())
    
    # Process based on file extension
    if ext == '.docx':
        # Extract text from Word document
        return docx2txt.process(file)
    elif ext == '.pdf':
        # Extract text from PDF
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        except Exception as e:
            text = f"Error extracting PDF content: {str(e)}"
        return text
    elif ext in ['.txt', '.md', '.rst']:
        # Read plain text files
        return file.getvalue().decode('utf-8')
    else:
        return f"Unsupported file format: {ext}"

def preprocess_text(text: str, language: str, remove_punctuation: bool = False, 
                   remove_spaces: bool = False, remove_english: bool = False,
                   remove_numbers: bool = False) -> str:
    """
    Preprocess text based on specified options.
    
    Args:
        text: Input text
        language: Language of the text ('Chinese' or 'English')
        remove_punctuation: Whether to remove punctuation
        remove_spaces: Whether to remove spaces
        remove_english: Whether to remove English letters (for Chinese text)
        remove_numbers: Whether to remove numbers
        
    Returns:
        str: Preprocessed text
    """
    processed_text = text
    
    # Apply text processing options
    if remove_punctuation:
        # Remove all punctuation
        processed_text = re.sub(r'[^\w\s]', '', processed_text)
    
    if language == "Chinese":
        if remove_english:
            # Remove all English letters for Chinese text
            processed_text = re.sub(r'[a-zA-Z]', '', processed_text)
    
    if remove_spaces:
        # Remove all spaces
        processed_text = re.sub(r'\s+', '', processed_text)
    
    if remove_numbers:
        # Remove all Arabic numerals
        processed_text = re.sub(r'[0-9]', '', processed_text)
        
    return processed_text

def tokenize_for_ngrams(text: str, language: str, word_level: bool = False) -> List[str]:
    """
    Tokenize text for N-gram analysis.
    
    Args:
        text: Input text
        language: Language of the text ('Chinese' or 'English')
        word_level: Whether to tokenize at word level (for English)
        
    Returns:
        List[str]: List of tokens
    """
    if language == "English" and word_level:
        # For English word-level N-grams, tokenize by words
        return text.split()
    else:
        # For character-level N-grams (default for Chinese and optional for English)
        return list(text)
