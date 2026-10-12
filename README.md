# 智扫通智能客服

基于 **LangChain ReAct Agent + RAG** 的扫地机器人售后助手：能做选购 / 故障 / 保养问答，也能按使用记录生成个性化报告。

仓库：[LangChain-Rag-Agent](https://github.com/ztr18552092590/LangChain-Rag-Agent)

## 功能

- **ReAct Agent**：通义千问 `qwen3-max` 自主规划工具调用
- **RAG**：Chroma + `text-embedding-v4`，按编号条目入库，只回原文；距离过大返回「知识库未覆盖」
- **意图分流**：规则判断「知识问答 / 使用报告」，中间件限制可用工具和调用顺序
- **动态提示词**：问答用客服 prompt；调用 `fill_context_for_report` 后切换为报告写手 prompt
- **会话记忆**：LangGraph SQLite checkpointer（`data/checkpoints.sqlite`），与 FAQ 向量库分开

## 技术栈

| 层级 | 选型 |
| --- | --- |
| 聊天模型 | 通义千问 `qwen3-max`（`ChatTongyi`） |
| 向量模型 | DashScope `text-embedding-v4` |
| Agent | LangChain `create_agent` + LangGraph |
| 向量库 | Chroma |
| 会话记忆 | `langgraph-checkpoint-sqlite` |
| 界面 | Streamlit |
| 配置 | YAML |

## 环境

- Python 3.10+
- 环境变量 **`DASHSCOPE_API_KEY`**（LangChain 会自动读取，业务代码不写死 Key）

```powershell
pip install streamlit langchain langchain-community langchain-chroma langgraph langgraph-checkpoint-sqlite dashscope pyyaml chromadb
```

## 启动

在项目根目录：

```powershell
# 1. 首次或知识库变更后，重建向量库（会调用 embedding）
#    若已有旧的按字数切分的索引，先删掉再导入
Remove-Item -Recurse -Force chroma_db, md5.text -ErrorAction SilentlyContinue
python -m rag.vector_store

# 2. 启动界面
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

## 目录

```text
app.py                 Streamlit 入口
agent/
  react_agent.py       Agent 组装、流式输出、checkpointer
  intent.py            问答 / 报告意图分流
  checkpointer.py      SQLite 会话记忆
  tools/               工具 + middleware（拦截、提示词切换、裁剪历史）
rag/
  splitter.py          按「1. 2. 3.」切完整知识条目
  vector_store.py      Chroma 入库 / 检索
  rag_service.py       只返回原文 + 距离过滤
prompts/               客服 prompt、报告 prompt
config/                rag.yml / chroma.yml / agent.yml / prompts.yml
data/                  FAQ 文本；external/records.csv 为模拟使用记录
```

`data/external` 不会进入 FAQ 向量库。会话 checkpoint 在 `data/checkpoints.sqlite`。

## 对话怎么走

**知识问答**（如「开机指示灯不亮怎么办」）

1. 意图 = 问答，只开放检索 / 定位 / 天气
2. `rag_summarize` 用短检索词查向量库
3. 距离 ≤ `max_distance` 则回原文，由 Agent 组织回答；否则「知识库未覆盖」，禁止编造

**使用报告**（如「生成我 2025 年 11 月的使用报告」）

1. 意图 = 报告
2. 顺序由中间件卡住：`get_user_id` → `get_current_month`（用户指定了月份则用指定值）→ `fill_context_for_report`（切换报告 prompt）→ `fetch_external_data`
3. 拿到记录后才允许再检索保养建议

同一浏览器会话内 user_id、城市固定；`get_current_month` 返回真实 `YYYY-MM`。演示报告时建议指定 CSV 中存在的月份（当前模拟数据多为 2025 年）。

## 配置

| 文件 | 常用项 |
| --- | --- |
| `config/rag.yml` | `chat_model_name`、`embedding_model_name` |
| `config/chroma.yml` | `k`（默认 6）、`max_distance`（默认 1.5，越大越容易召回） |
| `config/agent.yml` | `max_tool_calls`、`max_checkpoint_messages` |

召回太松：把 `max_distance` 调小；总报未覆盖：调大。改分块或知识文件后必须重建 `chroma_db`。

## 可试的问题

- 小户型有宠物适合什么扫地机器人？
- 开机指示灯不亮怎么办？
- 扫地机器人在我所在地区的气温下如何保养？
- 生成我 2025 年 11 月的使用报告
- （上一句之后）刚才说的那款要怎么保养？
