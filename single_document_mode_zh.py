"""
单文档分析模式
此模块提供分析单个文档中 N-gram 的功能。
"""
import streamlit as st
import pandas as pd
import re
import plotly.express as px
from collections import Counter
from typing import Dict, List, Tuple, Optional
import base64

from modules.document_processor import extract_text_from_file, preprocess_text, tokenize_for_ngrams

@st.cache_data
def generate_ngrams(tokens, n, word_level=False):
    """生成 n-gram 列表。
    
    Args:
        tokens: 分词后的文本列表
        n: n-gram 的大小
        word_level: 是否是词级分析
        
    Returns:
        List[str]: n-gram 列表
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        if word_level:
            # 词级分析时使用空格连接
            ngram = ' '.join(tokens[i:i+n])
        else:
            # 字符级分析时直接连接
            ngram = ''.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

@st.cache_data
def analyze_text(text, n, top_n, language="中文", remove_punctuation=True, 
                remove_spaces=True, remove_english=False, word_level=False, 
                remove_numbers=False):
    """
    分析文本以找出最频繁的 n-gram。
    
    参数:
        text: 要分析的文本
        n: n-gram 大小
        top_n: 返回的顶部 n-gram 数量
        language: 文本语言 ("中文" 或 "英文")
        remove_punctuation: 是否移除标点符号
        remove_spaces: 是否移除空格
        remove_english: 是否移除英文字母
        word_level: 是否进行词级分析而非字符级
        remove_numbers: 是否移除数字
        
    返回:
        包含分析结果的字典
    """
    # 预处理文本
    processed_text = preprocess_text(
        text, 
        language=language,
        remove_punctuation=remove_punctuation,
        remove_spaces=remove_spaces,
        remove_english=remove_english,
        remove_numbers=remove_numbers
    )
    
    # 分词
    tokens = tokenize_for_ngrams(processed_text, language=language, word_level=word_level)
    
    # 生成 n-gram
    ngrams = generate_ngrams(tokens, n, word_level=word_level)
    
    # 计算频率
    ngram_counts = Counter(ngrams)
    
    # 获取最常见的 n-gram
    top_ngrams = ngram_counts.most_common(top_n)
    
    return {
        "original_text": text,
        "processed_text": processed_text,
        "ngrams": top_ngrams,
        "total_ngrams": len(ngrams),
        "unique_ngrams": len(ngram_counts),
        "n_value": n
    }

def run_single_document_mode(language, n_value, top_n, remove_punctuation, remove_spaces, 
                            remove_english, word_level, remove_numbers, color_theme):
    """
    运行单文档分析模式。
    
    参数:
        language: 文本语言 ("中文" 或 "英文")
        n_value: n-gram 大小
        top_n: 返回的顶部 n-gram 数量
        remove_punctuation: 是否移除标点符号
        remove_spaces: 是否移除空格
        remove_english: 是否移除英文字母
        word_level: 是否进行词级分析而非字符级
        remove_numbers: 是否移除数字
        color_theme: 可视化的颜色主题
    """
    st.markdown("## 单文档 N-gram 分析")
    
    # 输入方法选择
    input_method = st.radio(
        "选择输入方式：",
        options=["文本输入", "文件上传"],
        index=0,
        help="选择直接输入文本或上传文件。"
    )
    
    if input_method == "文本输入":
        # 文本输入区域
        if language == "中文":
            placeholder = "请在此输入中文文本进行分析..."
        else:
            placeholder = "Enter English text here for analysis..."
            
        text_input = st.text_area(
            "",  # 移除标签，因为已经有上方的标题
            height=200,
            placeholder=placeholder,
            key="text_input_area"
        )
        
        # 分析按钮
        if st.button("分析文本", key="analyze_text_button"):
            if text_input.strip():
                with st.spinner("正在分析文本..."):
                    # 分析文本
                    results = analyze_text(
                        text_input,
                        n_value,
                        top_n,
                        language=language,
                        remove_punctuation=remove_punctuation,
                        remove_spaces=remove_spaces,
                        remove_english=remove_english,
                        word_level=word_level,
                        remove_numbers=remove_numbers
                    )
                    
                    # 显示结果
                    display_results(results, n_value, color_theme)
            else:
                st.error("请输入要分析的文本。")
    else:
        # 文件上传
        st.markdown("### 上传文档")
        st.markdown("上传文档以分析其 N-gram 模式。支持的格式：.txt，.md，.docx，.pdf")
        
        uploaded_file = st.file_uploader("上传文档", type=["txt", "md", "docx", "pdf"])
        
        if uploaded_file is not None:
            st.success(f"已上传文件：{uploaded_file.name}")
            
            # 分析按钮
            if st.button("分析文档", key="analyze_doc_button"):
                with st.spinner("正在分析文档..."):
                    # 提取文本
                    text = extract_text_from_file(uploaded_file, uploaded_file.name)
                    
                    # 分析文本
                    results = analyze_text(
                        text,
                        n_value,
                        top_n,
                        language=language,
                        remove_punctuation=remove_punctuation,
                        remove_spaces=remove_spaces,
                        remove_english=remove_english,
                        word_level=word_level,
                        remove_numbers=remove_numbers
                    )
                    
                    # 显示结果
                    display_results(results, n_value, color_theme)
        else:
            st.info("请上传文档以开始分析。")

def display_results(results, n_value, color_theme):
    """
    显示分析结果。
    
    参数:
        results: 分析结果字典
        n_value: n-gram 大小
        color_theme: 可视化的颜色主题
    """
    # 直接使用 Plotly 支持的标准颜色名称
    # 不需要颜色映射
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    # 显示原始文本和处理后的文本
    st.markdown("### 原始文本")
    st.text_area("", value=results["original_text"], height=100, disabled=True, key="original_text_area")
    
    st.markdown("### 处理后的文本")
    st.text_area("", value=results["processed_text"], height=100, disabled=True, key="processed_text_area")
    
    # 显示统计信息
    st.markdown("<h3 style='font-size: 1.3rem;'>文档统计信息</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("文本长度", len(results["processed_text"]))
    col2.metric("总 {}-gram 数量".format(results["n_value"]), results["total_ngrams"])
    col3.metric("不同的 {}-gram 数量".format(results["n_value"]), results["unique_ngrams"])
    
    # 添加对"唯一 N-gram 数量"的简要解释
    st.markdown("""
    <div style="font-size: 0.9rem; color: #4a5568; margin: 0.5rem 0 1rem 0;">
    <strong>不同的 N-gram 数量</strong>：指文本中出现的不同 N-gram 组合的总数，不考虑重复出现的频率。这是衡量文本多样性和复杂性的重要指标。
    </div>
    """, unsafe_allow_html=True)
    
    # 显示Top N-gram结果
    st.markdown(f"### Top {len(results['ngrams'])} {results['n_value']}-gram")
    
    # 准备数据框
    df = pd.DataFrame(results["ngrams"], columns=["N-gram", "频率"])
    
    # 计算百分比
    df["百分比"] = df["频率"] / results["total_ngrams"] * 100
    df["百分比"] = df["百分比"].round(2)
    df["百分比"] = df["百分比"].astype(str) + '%'
    
    # 创建条形图
    fig = px.bar(
        df,
        x="N-gram",
        y="频率",
        title=f"Top {len(results['ngrams'])} {results['n_value']}-gram"
        # 移除频率颜色映射
        # color="频率",
        # color_continuous_scale=color_theme
    )
    
    # 不再需要删除频率示意柱，因为没有使用颜色映射
    # fig.update_layout(coloraxis_showscale=False)
    
    # 更新布局
    fig.update_layout(
        xaxis_title=f"{results['n_value']}-gram",
        yaxis_title="频率",
        height=500
    )
    
    # 显示图表
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示数据表
    st.markdown("### 详细数据")
    st.dataframe(df, use_container_width=True)
    
    # 下载选项
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ngram_results.csv">下载 CSV 文件</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
