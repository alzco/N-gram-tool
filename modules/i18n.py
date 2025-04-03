"""
Internationalization Module
This module provides translations for the N-gram Analyzer application.
"""

# English translations
en = {
    # General
    "app_title": "N-gram Text Analyzer",
    "footer_text": "N-gram Text Analyzer | Developed for Academic Research",
    
    # Mode selection
    "mode_select_title": "Select Analysis Mode",
    "mode_select_label": "Choose analysis mode:",
    "single_doc_mode": "Single Document Analysis",
    "multi_doc_mode": "Multi-Document Comparison",
    "single_doc_desc": "Analyze N-grams in a single document. Input text directly or upload a file.",
    "multi_doc_desc": "Compare N-gram patterns across multiple documents. Upload multiple files to analyze similarities and differences.",
    
    # Options
    "options_title": "Analysis Options",
    "ngram_size_label": "N-gram Size:",
    "results_num_label": "Number of Results:",
    "language_select_title": "Language Selection",
    "language_select_label": "Select primary text language:",
    "chinese_option": "Chinese",
    "english_option": "English",
    "chinese_mode_desc": "Optimized for character-level analysis of Chinese text. Automatically removes punctuation and spaces by default for better results.",
    "english_mode_desc": "Defaults to word-level N-gram analysis, which is more meaningful for English text. Character-level analysis is also available.",
    
    # Text processing options
    "text_proc_title": "Text Processing Options",
    "remove_punct_label": "Remove Punctuation",
    "remove_punct_help": "Toggle to remove punctuation marks from the text before analysis.",
    "remove_spaces_label": "Remove Spaces",
    "remove_spaces_help": "Toggle to remove all spaces from the text before analysis.",
    "remove_english_label": "Remove English Letters",
    "remove_english_help": "Toggle to remove all English letters (a-z, A-Z) from the text before analysis.",
    "word_level_label": "Word-level N-grams",
    "word_level_help": "Toggle to analyze word-level N-grams instead of character-level N-grams.",
    "remove_numbers_label": "Remove Arabic Numerals",
    "remove_numbers_help": "Toggle to remove all Arabic numerals (0-9) from the text before analysis.",
    
    # Visualization options
    "viz_options_title": "Visualization Options",
    "color_theme_label": "Color Theme:",
    
    # Single document mode
    "single_doc_title": "Single Document N-gram Analysis",
    "input_method_label": "Select input method:",
    "text_input_option": "Text Input",
    "file_upload_option": "File Upload",
    "text_input_placeholder": "Enter English text here for analysis...",
    "analyze_text_button": "Analyze Text",
    "please_enter_text": "Please enter some text to analyze.",
    
    # File upload
    "upload_doc_title": "Upload a Document",
    "upload_doc_desc": "Upload a document to analyze its N-gram patterns. Supported formats: .txt, .md, .docx, .pdf",
    "upload_doc_label": "Upload a document",
    "uploaded_file": "Uploaded file:",
    "analyze_doc_button": "Analyze Document",
    "please_upload_doc": "Please upload a document to begin analysis.",
    
    # Multi-document mode
    "multi_doc_title": "Multi-Document N-gram Comparison",
    "upload_docs_title": "Upload Documents",
    "upload_docs_desc": "Upload multiple documents to compare their N-gram profiles. Supported formats: .txt, .md, .docx, .pdf",
    "upload_docs_label": "Upload documents for comparison",
    "docs_uploaded": "documents uploaded",
    "uploaded_docs_title": "Uploaded Documents:",
    "analyze_docs_button": "Analyze Documents",
    "please_upload_docs": "Please upload documents to begin analysis.",
    
    # Advanced analysis options
    "adv_analysis_title": "Advanced Analysis Options",
    "similarity_method_label": "Similarity Calculation Method:",
    "min_distinctive_freq_label": "Minimum Frequency for Distinctive N-grams:",
    "heatmap_color_label": "Similarity Heatmap Color Scheme:",
    
    # Results
    "analyzing_text": "Analyzing text...",
    "analyzing_doc": "Analyzing document...",
    "analyzing_docs": "Analyzing documents...",
    "analysis_results": "Analysis Results",
    "no_results": "No results found. Try adjusting your parameters or using a different text.",
    "top_ngrams_title": "Top {n} {size}-grams",
    "freq_viz_title": "Frequency Visualization",
    "export_results": "Export Results",
    "download_results_csv": "Download Results (CSV)",
    
    # Multi-document results
    "ngram_freq_comparison": "N-gram Frequency Comparison",
    "download_comparison_chart": "Download Comparison Chart",
    "similarity_matrix_title": "Document Similarity Matrix",
    "using_similarity_method": "Using {method} to compare document N-gram profiles:",
    "download_similarity_heatmap": "Download Similarity Heatmap",
    "distinctive_ngrams_title": "Distinctive N-grams",
    "distinctive_ngrams_desc": "N-grams that appear uniquely in each document (minimum frequency: {min_freq}):",
    "distinctive_ngrams_in": "Distinctive N-grams in {doc_name}",
    "no_distinctive_ngrams": "No distinctive N-grams found for this document.",
    "individual_doc_results": "Individual Document Results",
    "results_for_doc": "Results for {doc_name}",
    "download_csv": "Download CSV",
    "export_all_results": "Export All Results",
    "download_all_results": "Download All Results (CSV)",
    "download_similarity_matrix": "Download Similarity Matrix (CSV)"
}

# Chinese translations
zh = {
    # General
    "app_title": "N-gram 文本分析器",
    "footer_text": "N-gram 文本分析器 | 为学术研究开发",
    
    # Mode selection
    "mode_select_title": "选择分析模式",
    "mode_select_label": "选择分析模式：",
    "single_doc_mode": "单文档分析",
    "multi_doc_mode": "多文档比较",
    "single_doc_desc": "分析单个文档中的 N-gram。直接输入文本或上传文件。",
    "multi_doc_desc": "比较多个文档之间的 N-gram 模式。上传多个文件以分析相似性和差异。",
    
    # Options
    "options_title": "分析选项",
    "ngram_size_label": "N-gram 大小：",
    "results_num_label": "结果数量：",
    "language_select_title": "语言选择",
    "language_select_label": "选择主要文本语言：",
    "chinese_option": "中文",
    "english_option": "英文",
    "chinese_mode_desc": "针对中文文本的字符级分析进行了优化。默认自动删除标点符号和空格以获得更好的结果。",
    "english_mode_desc": "默认为词级 N-gram 分析，这对英文文本更有意义。也可以使用字符级分析。",
    
    # Text processing options
    "text_proc_title": "文本处理选项",
    "remove_punct_label": "去除标点符号",
    "remove_punct_help": "切换以在分析前从文本中删除标点符号。",
    "remove_spaces_label": "去除空格",
    "remove_spaces_help": "切换以在分析前从文本中删除所有空格。",
    "remove_english_label": "去除英文字母",
    "remove_english_help": "切换以在分析前从文本中删除所有英文字母（a-z, A-Z）。",
    "word_level_label": "词级 N-gram",
    "word_level_help": "切换以分析词级 N-gram 而不是字符级 N-gram。",
    "remove_numbers_label": "去除阿拉伯数字",
    "remove_numbers_help": "切换以在分析前从文本中删除所有阿拉伯数字（0-9）。",
    
    # Visualization options
    "viz_options_title": "可视化选项",
    "color_theme_label": "颜色主题：",
    
    # Single document mode
    "single_doc_title": "单文档 N-gram 分析",
    "input_method_label": "选择输入方式：",
    "text_input_option": "文本输入",
    "file_upload_option": "文件上传",
    "text_input_placeholder": "请在此输入中文文本进行分析...",
    "analyze_text_button": "分析文本",
    "please_enter_text": "请输入一些文本进行分析。",
    
    # File upload
    "upload_doc_title": "上传文档",
    "upload_doc_desc": "上传文档以分析其 N-gram 模式。支持的格式：.txt, .md, .docx, .pdf",
    "upload_doc_label": "上传文档",
    "uploaded_file": "已上传文件：",
    "analyze_doc_button": "分析文档",
    "please_upload_doc": "请上传文档开始分析。",
    
    # Multi-document mode
    "multi_doc_title": "多文档 N-gram 比较",
    "upload_docs_title": "上传文档",
    "upload_docs_desc": "上传多个文档以比较它们的 N-gram 模式。支持的格式：.txt, .md, .docx, .pdf",
    "upload_docs_label": "上传文档进行比较",
    "docs_uploaded": "个文档已上传",
    "uploaded_docs_title": "已上传文档：",
    "analyze_docs_button": "分析文档",
    "please_upload_docs": "请上传文档开始分析。",
    
    # Advanced analysis options
    "adv_analysis_title": "高级分析选项",
    "similarity_method_label": "相似度计算方法：",
    "min_distinctive_freq_label": "独特 N-gram 的最小频率：",
    "heatmap_color_label": "相似度热图配色方案：",
    
    # Results
    "analyzing_text": "正在分析文本...",
    "analyzing_doc": "正在分析文档...",
    "analyzing_docs": "正在分析文档...",
    "analysis_results": "分析结果",
    "no_results": "未找到结果。尝试调整参数或使用不同的文本。",
    "top_ngrams_title": "前 {n} 个 {size}-gram",
    "freq_viz_title": "频率可视化",
    "export_results": "导出结果",
    "download_results_csv": "下载结果 (CSV)",
    
    # Multi-document results
    "ngram_freq_comparison": "N-gram 频率比较",
    "download_comparison_chart": "下载比较图表",
    "similarity_matrix_title": "文档相似度矩阵",
    "using_similarity_method": "使用 {method} 比较文档 N-gram 模式：",
    "download_similarity_heatmap": "下载相似度热图",
    "distinctive_ngrams_title": "独特 N-gram",
    "distinctive_ngrams_desc": "在每个文档中唯一出现的 N-gram（最小频率：{min_freq}）：",
    "distinctive_ngrams_in": "{doc_name} 中的独特 N-gram",
    "no_distinctive_ngrams": "此文档未找到独特的 N-gram。",
    "individual_doc_results": "单个文档结果",
    "results_for_doc": "{doc_name} 的结果",
    "download_csv": "下载 CSV",
    "export_all_results": "导出所有结果",
    "download_all_results": "下载所有结果 (CSV)",
    "download_similarity_matrix": "下载相似度矩阵 (CSV)"
}

# Dictionary of available languages
languages = {
    "English": en,
    "中文": zh
}

def get_text(key, lang_dict, **kwargs):
    """
    Get translated text for a given key.
    
    Args:
        key: The translation key
        lang_dict: The language dictionary to use
        **kwargs: Format parameters for the translation
        
    Returns:
        str: The translated text
    """
    if key not in lang_dict:
        return f"[Missing: {key}]"
    
    text = lang_dict[key]
    
    # Apply formatting if kwargs are provided
    if kwargs:
        try:
            return text.format(**kwargs)
        except KeyError:
            return text
    
    return text
