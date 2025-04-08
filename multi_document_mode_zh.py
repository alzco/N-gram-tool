"""
多文档分析模式
此模块提供比较多个文档之间 N-gram 的功能。
"""
import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go  # 保留用于热图生成
from collections import Counter
from typing import Dict, List, Tuple, Optional
import base64

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
    """
    运行多文档比较模式。
    
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
        similarity_method: 相似度计算方法 ("cosine", "jaccard", "overlap")
        min_distinctive_freq: 独特 N-gram 的最小频率
        heatmap_color: 热图的颜色方案
    """
    # Matplotlib 和 Plotly 的颜色名称格式不同，需要进行转换
    matplotlib_color_map = {
        "blues": "Blues",
        "greens": "Greens",
        "oranges": "Oranges",
        "purples": "Purples",
        "teal": "YlGnBu",  # Matplotlib 没有 teal，使用类似的颜色
        "ylorbr": "YlOrBr",
        "ylgn": "YlGn",
        "greys": "Greys",
        "bugn": "BuGn",
        "bupu": "BuPu"
    }
    st.markdown("## 多文档 N-gram 比较")
    
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
    st.markdown("上传多个文档以比较它们的 N-gram 模式。支持的格式：.txt，.md，.docx，.pdf")
    
    uploaded_files = st.file_uploader(
        "上传文档进行比较", 
        type=["txt", "md", "docx", "pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"已上传 {len(uploaded_files)} 个文档")
        
        # 显示文件名
        st.markdown("### 文件名称")
        for i, file in enumerate(uploaded_files):
            st.text(f"{i+1}. {file.name}")
        
        # 分析按钮
        if st.button("分析文档", key="analyze_docs_button"):
            if len(uploaded_files) < 2:
                st.error("您需要至少上传两个文档进行比较。")
            else:
                with st.spinner("正在分析文档..."):
                    # 处理每个文档
                    documents = []
                    for file in uploaded_files:
                        # 提取文本
                        text = extract_text_from_file(file, file.name)
                        
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
                        
                        # 添加到文档列表
                        documents.append({
                            "name": file.name,
                            "original_text": text,
                            "processed_text": processed_text,
                            "ngram_counts": ngram_counts,
                            "top_ngrams": top_ngrams,
                            "total_ngrams": sum(ngram_counts.values()),
                            "unique_ngrams": len(ngram_counts)
                        })
                    
                    # 显示结果
                    if documents:
                        st.markdown('<div class="results-container">', unsafe_allow_html=True)
                        
                        # 文档统计信息
                        st.markdown("### 文档统计信息")
                        stats_data = []
                        for doc in documents:
                            stats_data.append({
                                "文档": doc["name"],
                                "文本长度": len(doc["processed_text"]),
                                "总 N-gram 数量": doc["total_ngrams"],
                                "唯一 N-gram 数量": doc["unique_ngrams"]
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # 相似度矩阵
                        st.markdown("### 文档相似度")
                        
                        # 准备 n-gram 计数器字典
                        doc_ngrams = {doc["name"]: doc["ngram_counts"] for doc in documents}
                        
                        # 计算相似度矩阵
                        similarity_matrix = calculate_document_similarity(doc_ngrams, method=similarity_method)
                        
                        # 显示相似度矩阵
                        # 将 Plotly 颜色名称转换为 Matplotlib 颜色名称
                        matplotlib_cmap = matplotlib_color_map.get(heatmap_color, "Blues")  # 默认使用 Blues
                        st.dataframe(similarity_matrix.style.background_gradient(cmap=matplotlib_cmap), use_container_width=True)
                        
                        # 生成热图
                        st.markdown("### 相似度热图")
                        
                        # 创建热图
                        fig = px.imshow(
                            similarity_matrix.values,
                            x=similarity_matrix.columns,
                            y=similarity_matrix.index,
                            color_continuous_scale=heatmap_color,
                            labels=dict(x="文档", y="文档", color="相似度"),
                            title="文档相似度热图"
                        )
                        
                        # 更新布局
                        fig.update_layout(
                            height=500,
                            xaxis=dict(side="bottom")
                        )
                        
                        # 显示热图
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 找出每个文档的独特 N-gram
                        st.markdown("### 文档独特的 N-gram")
                        
                        # 计算独特 N-gram
                        distinctive_ngrams = find_distinctive_ngrams(doc_ngrams, min_freq=min_distinctive_freq)
                        
                        # 为每个文档显示独特的 N-gram
                        tabs = st.tabs([doc["name"] for doc in documents])
                        
                        for i, tab in enumerate(tabs):
                            with tab:
                                doc_name = documents[i]["name"]
                                doc_distinctive = distinctive_ngrams.get(doc_name, {})
                                
                                if doc_distinctive:
                                    # 准备数据框
                                    df = pd.DataFrame({
                                        "N-gram": [item[0] for item in doc_distinctive],
                                        "频率": [item[1] for item in doc_distinctive]
                                    })
                                    
                                    # 创建条形图
                                    fig = px.bar(
                                        df,
                                        x="N-gram",
                                        y="频率",
                                        title=f"{documents[i]['name']} 的独特 {n_value}-gram",
                                        color="频率",
                                        color_continuous_scale=color_theme
                                    )
                                    
                                    # 删除频率示意柱
                                    fig.update_layout(coloraxis_showscale=False)
                                    
                                    # 更新布局
                                    fig.update_layout(
                                        xaxis_title=f"{n_value}-gram",
                                        yaxis_title="频率",
                                        height=400
                                    )
                                    
                                    # 显示图表
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 显示数据表
                                    st.dataframe(df, use_container_width=True)
                                else:
                                    st.info(f"未找到 {doc_name} 的独特 N-gram。")
                        
                        # 单个文档的N-gram分析结果
                        st.markdown("### 单个文档的 N-gram 分析")
                        doc_tabs = st.tabs([文档["name"] for 文档 in documents])
                        
                        for i, tab in enumerate(doc_tabs):
                            with tab:
                                doc_name = documents[i]["name"]
                                # 准备数据框
                                df = pd.DataFrame(documents[i]["top_ngrams"], columns=["N-gram", "频率"])
                                
                                # 计算百分比
                                df["百分比"] = df["频率"] / documents[i]["total_ngrams"] * 100
                                df["百分比"] = df["百分比"].round(2)
                                df["百分比"] = df["百分比"].astype(str) + '%'
                                
                                # 创建条形图
                                fig = px.bar(
                                    df,
                                    x="N-gram",
                                    y="频率",
                                    title=f"{doc_name} 的 Top {len(df)} {n_value}-gram",
                                    color="频率",
                                    color_continuous_scale=color_theme,
                                    height=400
                                )
                                
                                # 显示图表
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 显示数据表
                                st.dataframe(df, use_container_width=True)
                        
                        # 比较所有文档的Top N-gram
                        st.markdown(f"### Top {top_n} {n_value}-gram 比较")
                        
                        # 准备比较数据
                        comparison_data = []
                        for doc in documents:
                            for ngram, freq in doc["top_ngrams"]:
                                comparison_data.append({
                                    "文档": doc["name"],
                                    "N-gram": ngram,
                                    "频率": freq
                                })
                        
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # 创建分组条形图
                            fig = px.bar(
                                comparison_df,
                                x="N-gram",
                                y="频率",
                                color="文档",
                                barmode="group",
                                title=f"各文档 Top {top_n} {n_value}-gram 比较",
                                height=500
                            )
                            
                            # 更新布局
                            fig.update_layout(
                                xaxis_title=f"{n_value}-gram",
                                yaxis_title="频率"
                            )
                            
                            # 显示图表
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 显示数据表
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # 下载选项
                            csv = comparison_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="ngram_comparison.csv">下载 CSV 文件</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("请至少上传两个文档以开始比较。")
