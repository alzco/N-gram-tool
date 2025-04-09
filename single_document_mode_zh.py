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
from modules.ngram_analyzer import analyze_document_ngrams

def run_single_document_mode(language, n_value, top_n, remove_punctuation, remove_spaces, 
                            remove_english, word_level, remove_numbers, color_theme):
    """
    运行单文档分析模式。
    
    参数:
        language: 文本语言 ("中文" 或 "英文")
        n_value: n-gram 大小
        top_n: 返回的Top n-gram 数量
        remove_punctuation: 是否移除标点符号
        remove_spaces: 是否移除空格
        remove_english: 是否移除英文字母
        word_level: 是否进行词级分析而非字符级
        remove_numbers: 是否移除数字
        color_theme: 可视化的颜色主题
    """
    # 初始化会话状态变量
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    
    st.markdown("## 单文档 N-gram 分析")
    
    # 中文分词选项
    jieba_mode = None
    if language == "中文":
        st.markdown("### 中文分词设置")
        use_jieba = st.checkbox("启用结巴分词", value=False, help="使用结巴分词库进行中文分词")
        
        if use_jieba:
            jieba_mode = st.radio(
                "选择分词模式",
                options=["精确模式", "全模式"],
                index=0,
                help="精确模式尝试将句子最精确地切开，全模式将所有可能的词语都分割出来"
            )
            
            # 如果使用结巴分词，则强制设置为词级分析
            word_level = True
            st.info("使用结巴分词时，将自动启用词级分析。若选择结巴分词，建议N-gram大小设置为1。")
    
    # 文件上传
    st.markdown("### 上传文档")
    st.markdown("上传文档进行N-gram分析。支持的格式：.txt，.md，.docx，.pdf。上传的文档不会被Streamlit Cloud永久保存，它们只在用户会话期间存在于内存中，会话结束后就会被删除。")
    
    uploaded_file = st.file_uploader("上传文档", type=["txt", "md", "docx", "pdf"])
    
    if uploaded_file:
        # 分析按钮
        analyze_button = st.button("分析文档", key="analyze_button")
        
        # 如果点击了分析按钮或者会话状态中已有分析结果
        if analyze_button or st.session_state.analysis_results is not None:
            # 如果点击了分析按钮，重新分析文档
            if analyze_button:
                with st.spinner("正在分析文档..."):
                    # 提取文本
                    text = extract_text_from_file(uploaded_file, uploaded_file.name)
                    
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
                    tokens = tokenize_for_ngrams(processed_text, language=language, word_level=word_level, jieba_mode=jieba_mode)
                    
                    # 分析 n-gram
                    ngram_counts = Counter()
                    if len(tokens) >= n_value:
                        ngrams = []
                        for i in range(len(tokens) - n_value + 1):
                            if word_level:
                                # 词级分析时使用空格连接
                                ngram = ' '.join(tokens[i:i+n_value])
                            else:
                                # 字符级分析时直接连接
                                ngram = ''.join(tokens[i:i+n_value])
                            ngrams.append(ngram)
                        ngram_counts = Counter(ngrams)
                    
                    # 获取Top n-gram
                    top_ngrams = ngram_counts.most_common(top_n)
                    
                    # 保存分析结果到会话状态
                    st.session_state.analysis_results = {
                        "original_text": text,
                        "processed_text": processed_text,
                        "ngram_counts": ngram_counts,
                        "top_ngrams": top_ngrams,
                        "total_ngrams": sum(ngram_counts.values()),
                        "unique_ngrams": len(ngram_counts)
                    }
                    
                    # 清除之前的搜索结果
                    st.session_state.search_results = None
            
            # 显示分析结果
            display_results(st.session_state.analysis_results, n_value, color_theme)
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
    if results is None:
        return
    
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    # 显示Top N-gram
    st.markdown("### Top N-gram")
    
    # 准备数据框
    df = pd.DataFrame(results["top_ngrams"], columns=["N-gram", "频率"])
    
    # 计算百分比
    df["百分比"] = df["频率"] / results["total_ngrams"] * 100
    df["百分比"] = df["百分比"].round(2)
    df["百分比"] = df["百分比"].astype(str) + '%'
    
    # 创建条形图
    fig = px.bar(
        df,
        x="N-gram",
        y="频率",
        title=f"Top {len(df)} {n_value}-gram",
        color="频率",
        color_continuous_scale=color_theme,
        height=400
    )
    
    # 删除频率示意柱
    fig.update_layout(coloraxis_showscale=False)
    
    # 更新布局
    fig.update_layout(
        xaxis_title=f"{n_value}-gram",
        yaxis_title="频率"
    )
    
    # 显示图表
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示数据表
    st.dataframe(df, use_container_width=True)
    
    # 下载选项
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ngram_analysis.csv">下载 CSV 文件</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # 显示原始文本和处理后的文本
    st.markdown("### 原始文本")
    st.text_area("原始文本", results["original_text"], height=100, disabled=True, key="original_text_area")
    
    st.markdown("### 处理后的文本")
    st.text_area("处理后的文本", results["processed_text"], height=100, disabled=True, key="processed_text_area")
    
    # 显示统计信息
    st.markdown("<h3 style='font-size: 1.3rem;'>文档统计信息</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("文本长度", len(results["processed_text"]))
    
    with col2:
        st.metric("总 N-gram 数量", results["total_ngrams"])
    
    with col3:
        st.metric("唯一 N-gram 数量", results["unique_ngrams"])
    
    # 添加N-gram上下文检索功能
    st.markdown("### N-gram 上下文检索")
    st.markdown("输入要检索的N-gram，查看其在文档中的上下文。")
    
    # 创建三列布局
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_ngram = st.text_input("输入要检索的N-gram", key="search_input")
    
    with col2:
        context_window = st.number_input("上下文窗口大小", min_value=5, max_value=50, value=20, key="context_window")
    
    with col3:
        search_clicked = st.button("检索N-gram", key="search_button")
    
    # 检索N-gram
    if search_clicked and search_ngram:
        # 获取处理后的文本
        processed_text = st.session_state.analysis_results["processed_text"]
        
        # 在文本中查找N-gram
        contexts = []
        start_idx = 0
        
        while True:
            # 查找下一个匹配项
            pos = processed_text.find(search_ngram, start_idx)
            if pos == -1:
                break
                
            # 计算上下文窗口
            context_start = max(0, pos - context_window)
            context_end = min(len(processed_text), pos + len(search_ngram) + context_window)
            
            # 提取上下文
            left_context = processed_text[context_start:pos]
            right_context = processed_text[pos+len(search_ngram):context_end]
            
            # 添加到结果列表
            contexts.append({
                'left': left_context,
                'ngram': search_ngram,
                'right': right_context,
                'position': pos
            })
            
            # 更新下一次搜索的起始位置
            start_idx = pos + len(search_ngram)
        
        # 将检索结果保存到会话状态
        results_data = []
        for i, context in enumerate(contexts):
            # 创建上下文预览
            results_data.append({
                "上下文": f"...{context['left']} **{context['ngram']}** {context['right']}..."
            })
        
        st.session_state.search_results = results_data
    
    # 显示检索结果（如果有）
    if st.session_state.search_results is not None and len(st.session_state.search_results) > 0:
        result_count = len(st.session_state.search_results)
        st.markdown(f"#### 检索结果 (共{result_count}个结果)")
        st.dataframe(pd.DataFrame(st.session_state.search_results), use_container_width=True)
    elif search_clicked and search_ngram:
        st.info(f"未找到匹配的N-gram: '{search_ngram}'")
    
    st.markdown('</div>', unsafe_allow_html=True)
