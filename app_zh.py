"""
N-gram 文本分析器 - Streamlit 应用程序
此应用程序结合了单文档和多文档 N-gram 分析功能。
"""
import streamlit as st
import pandas as pd
import re
import plotly.express as px  # 保留 plotly 用于可视化
from collections import Counter
from typing import Dict, List, Tuple, Optional
import os

# 导入自定义模块
from modules.document_processor import extract_text_from_file, preprocess_text, tokenize_for_ngrams
from modules.ngram_analyzer import (
    analyze_document_ngrams, 
    calculate_document_similarity, 
    find_distinctive_ngrams,
    generate_similarity_heatmap,
    generate_ngram_comparison_chart
)

# 设置页面配置
st.set_page_config(
    page_title="N-gram 文本分析器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS，现代学术风格
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

# 应用标题
st.title("N-gram 文本分析器")

# 模式选择
st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
st.markdown("### 选择分析模式")
analysis_mode = st.radio(
    "选择分析模式：",
    options=["单文档分析", "多文档比较"],
    index=0,
    horizontal=True,
    help="选择是分析单个文档还是比较多个文档。"
)

if analysis_mode == "单文档分析":
    st.markdown("""
    <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
    分析单个文档中的 N-gram。直接输入文本或上传文件。
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
    对多个文档进行N-gram分析。上传多个文件以分析相似性和差异。
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 创建主列和侧边栏
col1 = st.container()
col2 = st.sidebar

# 侧边栏选项
with col2:
    st.markdown("### 分析选项")
    
    # 输入参数
    n_value = st.slider(
        "N-gram 大小：",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        help="N-gram 中的字符数量。例如，2-gram（二元组）分析两个连续字符的序列。"
    )
    
    top_n = st.slider(
        "结果数量：",
        min_value=1,
        max_value=100,
        value=10,
        step=5,
        help="要显示的最频繁 N-gram 的数量。"
    )
    
    # 语言选择
    st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
    st.markdown("##### 语言选项")
    
    language = st.radio(
        "选择主要文本语言：",
        options=["中文", "英文"],
        index=0,
        help="选择您要分析的文本的主要语言。这将优化处理和分析。"
    )
    
    if language == "中文":
        st.markdown("""
        **中文模式**：针对中文文本的字符级分析进行了优化。
        默认自动删除标点符号和空格以获得更好的结果。
        """)
    else:
        st.markdown("""
        **英文模式**：默认为词级 N-gram 分析，这对英文文本更有意义。
        也可以进行字符级分析。
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 文本处理选项
    st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
    st.markdown("##### 文本处理选项")
    
    remove_punctuation = st.toggle(
        "去除标点符号",
        value=(language == "中文"),
        help="切换以在分析前从文本中删除标点符号。"
    )
    
    remove_spaces = st.toggle(
        "去除空格",
        value=(language == "中文"),
        help="切换以在分析前从文本中删除所有空格。"
    )
    
    # 只在英文模式下显示词级N-gram选项
    if language == "英文":
        word_level = st.toggle(
            "词级 N-gram",
            value=True,
            help="切换以分析词级 N-gram 而不是字符级 N-gram。"
        )
    else:
        word_level = False
    
    if language == "中文":
        remove_english = st.toggle(
            "去除英文字母",
            value=False,
            help="切换以在分析前从文本中删除所有英文字母（a-z，A-Z）。"
        )
    else:
        remove_english = False
    
    # 对中文和英文模式都显示去除阿拉伯数字选项
    remove_numbers = st.toggle(
        "去除阿拉伯数字",
        value=False,
        help="切换以在分析前从文本中删除所有阿拉伯数字（0-9）。"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 可视化选项
    st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
    st.markdown("##### 可视化选项")
    
    color_theme = st.selectbox(
        "颜色主题：",
        options=[
            # Plotly 支持的标准颜色
            "blues", "greens", "oranges", "purples", "teal",
            "ylorbr", "ylgn", "greys", "bugn", "bupu"
        ],
        index=0,
        help="选择可视化的颜色主题。使用 Plotly 支持的标准颜色。"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 多文档特定选项
    if analysis_mode == "多文档比较":
        st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
        st.markdown("##### 高级分析选项")
        
        similarity_method = st.selectbox(
            "相似度计算方法：",
            options=["余弦相似度", "Jaccard 相似度", "重叠系数"],
            index=0,
            help="用于计算文档之间相似性的方法。"
        )
        
        # 显示相似度计算方法的详细解释
        if similarity_method == "余弦相似度":
            st.markdown("""
            <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
            <strong>余弦相似度</strong>测量两个向量之间的夹角余弦值。它考虑了向量的方向而非仅仅是大小，非常适合比较文本的语义相似性。范围从0（完全不相似）到1（完全相同）。
            </div>
            """, unsafe_allow_html=True)
        elif similarity_method == "Jaccard 相似度":
            st.markdown("""
            <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
            <strong>Jaccard 相似度</strong>测量两个集合的交集大小与并集大小的比值。它专注于共有元素，忽略频率信息，非常适合比较文档的内容重叠度。范围从0（无共同元素）到1（完全相同）。
            </div>
            """, unsafe_allow_html=True)
        else:  # 重叠系数
            st.markdown("""
            <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
            <strong>重叠系数</strong>考虑两个集合的交集与较小集合大小的比值。它特别适合比较不同大小的文档，因为它不会因为文档长度差异而惩罚相似度。范围从0（无共同元素）到1（小集合是大集合的子集）。
            </div>
            """, unsafe_allow_html=True)
        
        # 添加空白区域
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
        
        min_distinctive_freq = st.slider(
            "独特 N-gram 的最小频率：",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="将 N-gram 视为文档特有的最小频率。"
        )
        
        # 热图颜色选择
        heatmap_color = st.selectbox(
            "相似度热图配色方案：",
            options=[
                # Plotly 支持的标准颜色
                "blues", "greens", "oranges", "purples", "teal",
                # 更多 Plotly 颜色
                "ylorbr", "ylgn", "greys", "bugn", "bupu"
            ],
            index=0,
            help="相似度热图的配色方案。使用 Plotly 支持的标准颜色。"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# 主区域内容基于所选模式
with col1:
    if analysis_mode == "单文档分析":
        # 导入中文版单文档分析函数
        from single_document_mode_zh import run_single_document_mode
        
        # 运行单文档模式
        run_single_document_mode(
            language=language,
            n_value=n_value,
            top_n=top_n,
            remove_punctuation=remove_punctuation,
            remove_spaces=remove_spaces,
            remove_english=remove_english,
            word_level=word_level,
            remove_numbers=remove_numbers,
            color_theme=color_theme
        )
    else:  # 多文档比较
        # 导入中文版多文档分析函数
        from multi_document_mode_zh import run_multi_document_mode
        
        # 将中文选项转换为函数所需的格式
        similarity_method_map = {
            "余弦相似度": "cosine",
            "Jaccard 相似度": "jaccard",
            "重叠系数": "overlap"
        }
        
        # 运行多文档模式
        run_multi_document_mode(
            language=language,
            n_value=n_value,
            top_n=top_n,
            remove_punctuation=remove_punctuation,
            remove_spaces=remove_spaces,
            remove_english=remove_english,
            word_level=word_level,
            remove_numbers=remove_numbers,
            color_theme=color_theme,
            similarity_method=similarity_method_map[similarity_method],
            min_distinctive_freq=min_distinctive_freq,
            heatmap_color=heatmap_color
        )

# 页脚
st.markdown("---")
st.markdown("N-gram 文本分析器", unsafe_allow_html=True)
