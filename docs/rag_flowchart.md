# RAG 流程图

## 流程图 (Flowchart)

```mermaid
flowchart TD
    A[开始] --> B[1. 提取文本<br/>extract_text_from_markdown]
    B --> C[2. 分割文本<br/>chunk_text]
    C --> D[3. 构建向量模型<br/>EmbeddingModel]
    D --> E[4. 构建知识库向量集<br/>create_embeddings]
    E --> F[5. 构建问题向量<br/>create_embeddings]
    F --> G[6. 向量相似度检索<br/>semantic_search]
    G --> H[7. 调用LLM模型生成回答<br/>query_llm]
    H --> I[输出结果]
    
    %% 样式定义
    classDef stepClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef dataClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef modelClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    %% 应用样式
    class B,C stepClass
    class D,F,G modelClass
    class E dataClass
```

## 时序图 (Sequence Diagram)

```mermaid
sequenceDiagram
    participant User as 用户
    participant App as 主程序
    participant FileUtils as 文件工具
    participant Embedding as 嵌入模型
    participant Search as 搜索模块
    participant LLM as LLM模型
    
    User->>App: 输入查询问题
    App->>FileUtils: 1. 提取西游记文本
    FileUtils-->>App: 返回文本内容
    
    App->>App: 2. 分割文本为chunks
    App->>Embedding: 3. 初始化嵌入模型
    
    App->>Embedding: 4. 构建知识库向量集
    Embedding-->>App: 返回知识库嵌入向量
    
    App->>Embedding: 5. 构建问题向量
    Embedding-->>App: 返回查询嵌入向量
    
    App->>Search: 6. 向量相似度检索
    Search->>Search: 计算余弦相似度
    Search->>Search: 排序并选择top-k结果
    Search-->>App: 返回相关文本块及分数
    
    App->>LLM: 7. 调用LLM生成回答
    Note over App,LLM: 构建包含上下文的提示词
    LLM-->>App: 返回生成的答案
    
    App-->>User: 输出最终结果
```

## 详细说明

### 流程图说明
1. **提取文本**: 从 `data/xiyouji.md` 文件中读取西游记文本内容
2. **分割文本**: 将长文本按固定大小和重叠度分割成多个文本块
3. **构建向量模型**: 初始化 Sentence Transformers 嵌入模型
4. **构建知识库向量集**: 将所有文本块转换为向量表示
5. **构建问题向量**: 将用户查询转换为向量表示
6. **向量相似度检索**: 计算查询向量与知识库向量的相似度，返回最相关的文本块
7. **调用LLM模型**: 将相关文本块作为上下文，生成最终答案

### 时序图说明
- 展示了各个组件之间的交互顺序
- 突出了数据流向和处理步骤
- 显示了异步处理和同步返回的关系
