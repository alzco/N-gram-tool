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
    # 导入pandas
    import pandas as pd
    import re
    
    # 初始化会话状态变量
    if 'multi_documents' not in st.session_state:
        st.session_state.multi_documents = None
    if 'multi_search_results' not in st.session_state:
        st.session_state.multi_search_results = None
        
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
    st.markdown("上传多个文档以比较它们的 N-gram 模式。支持的格式：.txt，.md，.docx，.pdf。上传的文档不会被Streamlit Cloud永久保存，它们只在用户会话期间存在于内存中，会话结束后就会被删除。")
    
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
        analyze_button = st.button("分析文档", key="analyze_docs_button")
        
        # 如果点击了分析按钮或者会话状态中已有文档
        if (analyze_button and len(uploaded_files) >= 2) or st.session_state.multi_documents is not None:
            # 如果点击了分析按钮，重新分析文档
            if analyze_button and len(uploaded_files) >= 2:
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
                    
                    # 将文档保存到会话状态
                    st.session_state.multi_documents = documents
            
            # 使用会话状态中的文档
            documents = st.session_state.multi_documents
            
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
                
                # 显示图表
                st.plotly_chart(fig, use_container_width=True, key="heatmap_chart")
                
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
                            # 限制显示的独特N-gram数量为top_n
                            limited_distinctive = doc_distinctive[:top_n] if len(doc_distinctive) > top_n else doc_distinctive
                            df = pd.DataFrame({
                                "N-gram": [item[0] for item in limited_distinctive],
                                "频率": [item[1] for item in limited_distinctive]
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
                            st.plotly_chart(fig, use_container_width=True, key=f"distinctive_chart_{i}")
                            
                            # 显示数据表
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info(f"未找到 {doc_name} 的独特 N-gram。")
                
                # 单个文档的N-gram分析结果
                st.markdown("### 单个文档的 N-gram 分析")
                doc_tabs = st.tabs([doc["name"] for doc in documents])
                
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
                        st.plotly_chart(fig, use_container_width=True, key=f"doc_chart_{i}")
                        
                        # 显示数据表
                        st.dataframe(df, use_container_width=True)
                
                # 比较所有文档的Top N-gram - 简化版本，只保留比较图
                st.markdown(f"### Top {top_n} {n_value}-gram 比较")
                
                # 准备所有文档的比较数据
                all_comparison_data = []
                for doc in documents:
                    for ngram, freq in doc["top_ngrams"]:
                        all_comparison_data.append({
                            "文档": doc["name"],
                            "N-gram": ngram,
                            "频率": freq
                        })
                
                # 创建分组条形图
                if all_comparison_data:
                    all_comparison_df = pd.DataFrame(all_comparison_data)
                    
                    fig = px.bar(
                        all_comparison_df,
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
                    st.plotly_chart(fig, use_container_width=True, key="all_docs_comparison_chart")
                    
                    # 下载选项
                    csv = all_comparison_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="ngram_comparison.csv">下载 CSV 文件</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                # 添加N-gram上下文检索功能
                st.markdown("### N-gram 上下文检索")
                st.markdown("输入要检索的N-gram，查看其在文档中的上下文。")
                
                # 创建四列布局，确保对齐
                search_cols = st.columns([2, 1, 1, 1])
                
                with search_cols[0]:
                    search_ngram = st.text_input("输入要检索的N-gram", key="multi_search_input")
                
                with search_cols[1]:
                    context_size = st.number_input("上下文窗口大小", min_value=5, max_value=50, value=20, key="multi_context_size")
                
                with search_cols[2]:
                    search_doc = st.selectbox(
                        "选择要检索的文档",
                        ["所有文档"] + [doc["name"] for doc in documents],
                        key="multi_search_doc"
                    )
                
                with search_cols[3]:
                    search_button = st.button("检索N-gram", key="multi_search_button")
                
                # 检索N-gram
                if search_button and search_ngram:
                    # 初始化结果列表
                    all_contexts = []
                    
                    # 确定要搜索的文档
                    docs_to_search = documents if search_doc == "所有文档" else [doc for doc in documents if doc["name"] == search_doc]
                    
                    for doc in docs_to_search:
                        # 在原始文本中搜索N-gram
                        original_text = doc["original_text"]
                        doc_name = doc["name"]
                        
                        # 使用正则表达式查找所有匹配项
                        pattern = re.escape(search_ngram)
                        matches = list(re.finditer(pattern, original_text, re.IGNORECASE))
                        
                        # 提取每个匹配项的上下文
                        for match in matches:
                            start_pos = match.start()
                            end_pos = match.end()
                            
                            # 计算上下文范围
                            left_start = max(0, start_pos - context_size)
                            right_end = min(len(original_text), end_pos + context_size)
                            
                            # 提取上下文
                            left_context = original_text[left_start:start_pos]
                            right_context = original_text[end_pos:right_end]
                            
                            # 添加到结果列表
                            all_contexts.append({
                                "document": doc_name,
                                "ngram": search_ngram,
                                "left": left_context,
                                "right": right_context
                            })
                    
                    # 将检索结果按文档分组
                    results_by_doc = {}
                    for context in all_contexts:
                        doc_name = context['document']
                        if doc_name not in results_by_doc:
                            results_by_doc[doc_name] = []
                        
                        # 创建上下文预览
                        results_by_doc[doc_name].append({
                            "上下文": f"...{context['left']} **{context['ngram']}** {context['right']}..."
                        })
                    
                    # 保存到会话状态
                    st.session_state.multi_search_results = results_by_doc
                
                # 显示检索结果
                if st.session_state.multi_search_results and len(st.session_state.multi_search_results) > 0:
                    # 确保结果字典不为空
                    total_results = sum(len(results) for results in st.session_state.multi_search_results.values() if results)
                    st.markdown(f"#### 检索结果 (共{total_results}个结果)")
                    
                    # 创建文档标签页
                    if len(st.session_state.multi_search_results) > 1:
                        # 确保所有键都是有效的，并且结果不为空
                        valid_docs = [doc for doc, results in st.session_state.multi_search_results.items() if results]
                        
                        if valid_docs:
                            search_tabs = st.tabs(["All Documents"] + valid_docs)
                            
                            # 所有文档标签页
                            with search_tabs[0]:
                                # 合并所有文档的结果
                                all_results_data = []
                                for doc_name in valid_docs:
                                    results = st.session_state.multi_search_results.get(doc_name, [])
                                    for result in results:
                                        if isinstance(result, dict) and "上下文" in result:
                                            all_results_data.append({
                                                "文档": doc_name,
                                                "上下文": result["上下文"]
                                            })
                                
                                # 显示所有结果
                                st.markdown(f"所有文档中共找到 {len(all_results_data)} 个结果")
                                if all_results_data:
                                    st.dataframe(pd.DataFrame(all_results_data), use_container_width=True)
                            
                            # 单独文档标签页
                            for i, tab in enumerate(search_tabs[1:]):
                                with tab:
                                    doc_name = valid_docs[i]
                                    doc_results = st.session_state.multi_search_results.get(doc_name, [])
                                    
                                    # 显示当前文档的结果
                                    st.markdown(f"在 {doc_name} 中找到 {len(doc_results)} 个结果")
                                    if doc_results:
                                        st.dataframe(pd.DataFrame(doc_results), use_container_width=True)
                        else:
                            st.info("未找到有效的检索结果")
                    elif len(st.session_state.multi_search_results) == 1:
                        # 只有一个文档的情况
                        doc_name = list(st.session_state.multi_search_results.keys())[0]
                        doc_results = st.session_state.multi_search_results.get(doc_name, [])
                        
                        # 显示结果
                        st.markdown(f"在 {doc_name} 中找到 {len(doc_results)} 个结果")
                        if doc_results:
                            st.dataframe(pd.DataFrame(doc_results), use_container_width=True)
                    else:
                        st.info("未找到检索结果")
                elif search_button and search_ngram:
                    st.info(f"未找到匹配的N-gram: '{search_ngram}'")
                
                st.markdown('</div>', unsafe_allow_html=True)
            elif analyze_button and len(uploaded_files) < 2:
                st.error("您需要至少上传两个文档进行比较。")
    else:
        st.info("请至少上传两个文档以开始比较。")
