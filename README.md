# N-gram 文本分析工具 (N-gram Text Analysis Tool)

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44.0-red)

一个强大的基于 Streamlit 的 N-gram 文本分析工具，支持单文档分析和多文档比较。该工具为文本分析和文档相似度计算提供了直观的界面和强大的功能。

## 功能特点 (Features)

- **多格式支持**: 处理 TXT, DOCX, PDF, MD 等多种文件格式
- **多语言支持**: 支持中文和英文文本分析
- **单文档分析**: 详细的 N-gram 频率分析和可视化
- **多文档比较**: 计算文档间的相似度并生成热图，分析文档独特的 N-gram，Top N-gram 比较
- **N-gram 上下文检索**: 查看 N-gram 在文档中的上下文
- **高级可视化**: 使用 Plotly 生成交互式图表
- **数据导出**: 支持将分析结果导出为 CSV 格式
- **文本预处理**: 支持多种文本清洗和预处理选项

## 在线使用地址 (Online Usage)

https://n-gram-tool.streamlit.app/

支持的浏览器：
- Google Chrome
- Firefox
- Microsoft Edge
- Safari

## 本地安装 (Local Installation)

```bash
# 克隆仓库
git clone https://github.com/alzco/N-gram-tool.git
cd N-gram-tool

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 本地使用方法 (Local Usage)

```bash
# 启动中文版应用
streamlit run app_zh.py
```

然后在浏览器中访问 http://localhost:8501

## 主要功能 (Main Features)

### 单文档模式 (Single Document Mode):

- 上传文档并选择 N-gram 的大小 (1-20)
- 选择主要文本语言（中文/英文）
- 设置文本预处理选项
- 查看频率最高的 N-gram 和详细统计数据
- N-gram 上下文检索
- 导出结果为 PNG 图片或 CSV 格式

### 多文档模式 (Multi-Document Mode):

- 上传多个文档进行比较
- 计算文档间的相似度矩阵
- 生成相似度热图
- 找出文档间的共同和独特 N-gram
- Top N-gram 比较
- N-gram 上下文检索
- 导出结果为 PNG 图片或 CSV 格式

## 注意事项 (Notes)

- 对于大文件，处理可能需要一些时间
- 确保上传的文件格式正确
- 如果更改了配置，需要重新启动 Streamlit 应用程序

# 项目结构 (Project Structure)

```
N-gram-tool/
├── app_zh.py                # 中文版主应用程序
├── single_document_mode_zh.py # 中文版单文档分析模式
├── multi_document_mode_zh.py  # 中文版多文档比较模式
├── modules/                 # 核心功能模块
│   ├── document_processor.py  # 文档处理功能
│   └── ngram_analyzer.py      # N-gram 分析核心功能
├── .streamlit/               # Streamlit 配置
├── requirements.txt          # 项目依赖
└── README.md                 # 项目文档
```

# 关于 N-grams (About N-grams)

N-gram 是从给定文本样本中提取的 n 个连续项的序列。它是自然语言处理中的一种重要技术，广泛应用于文本分类、语言建模、机器翻译等领域。通过统计文本中的 N-gram词频，可以了解文本的特征。

例子:
英文单词 "hello" 的字符级 2-gram (二元组): "he", "el", "ll", "lo"
中文短语 "你好世界" 的字符级 2-gram: "你好", "好世", "世界"

唯一 N-gram 数量：指文本中出现的不同 N-gram 组合的总数，不考虑重复出现的频率。这是衡量文本多样性和复杂性的重要指标。


