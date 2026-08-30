# 智扫通 · 扫地机器人智能客服

基于 **LangChain ReAct Agent** 与 **RAG 检索增强** 的扫地机器人智能客服 Demo。用户可通过 Streamlit 对话界面咨询保养、选购、故障排除等问题，也可生成个人使用报告。

## 功能

- **专业问答**：从知识库检索维护保养、选购指南、故障排除与 FAQ，再由大模型总结作答
- **场景化建议**：结合用户位置与天气（当前为演示用 mock）给出环境适配建议
- **个人使用报告**：按用户 ID 与月份读取模拟使用记录，输出 Markdown 报告
- **流式对话**：Streamlit 聊天界面逐字输出回复

## 技术栈

| 类别 | 技术 |
|------|------|
| 界面 | Streamlit |
| Agent | LangChain `create_agent`（ReAct + 工具调用） |
| 知识库 | Chroma + 通义千问 Embedding |
| 大模型 | 通义千问（DashScope）`qwen3-max` |
| 文档解析 | TXT / PDF（pypdf） |

## 项目结构

```
├── app.py                 # Streamlit 入口
├── agent/                 # ReAct Agent 与工具
├── rag/                   # 向量入库与 RAG 总结
├── model/                 # Chat / Embedding 工厂
├── config/                # YAML 配置
├── prompts/               # 系统提示词与 RAG / 报告提示词
├── data/                  # 知识库源文件与模拟使用记录
├── utils/                 # 配置、路径、日志、文档加载
└── chroma_db/             # 向量库持久化（本地生成，不入库）
```

## 环境要求

- Python 3.10+（推荐 3.12）
- [阿里云 DashScope API Key](https://dashscope.console.aliyun.com/)（通义千问）

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 配置 API Key

在终端中设置环境变量（不要把密钥写进代码或提交到 Git）：

```powershell
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "sk-你的密钥"
```

```bash
# macOS / Linux
export DASHSCOPE_API_KEY="sk-你的密钥"
```

聊天模型与向量模型名称见 `config/rag.yml`（默认 `qwen3-max`、`text-embedding-v4`）。

### 4. 构建知识库

将 `.txt` / `.pdf` 放到 `data/` **目录根下**（不会递归扫描子目录，`data/external/` 不会进入向量库）。

在**项目根目录**执行：

```bash
python -m rag.vector_store
```

首次或更新文档后都需要执行。向量数据会写入 `chroma_db/`。

### 5. 启动 Web 界面

Windows 上 Streamlit 默认端口 **8501** 常被 Hyper-V / WSL 排除占用，会出现 `WinError 10013`。请改用其他端口：

```bash
streamlit run app.py --server.port 8000
```

浏览器打开：http://localhost:8000

### 可选：命令行试跑

```bash
# Agent（无界面）
python -m agent.react_agent

# 仅测试 RAG 总结
python -m rag.rag_service
```

## 对话示例

- 「扫地机器人滤网多久换一次？」
- 「我所在地区现在这种天气，扫地机该怎么保养？」
- 「帮我生成一份本月的个人使用报告」

报告类问题会按固定流程调用：获取用户 ID → 获取月份 → `fill_context_for_report` → `fetch_external_data`（必要时再检索知识库）。

## 配置说明

| 文件 | 作用 |
|------|------|
| `config/rag.yml` | 聊天模型、Embedding 模型 |
| `config/chroma.yml` | 向量库路径、检索条数 `k`、切分参数、知识库目录 |
| `config/agent.yml` | 外部使用记录 CSV 路径 |
| `config/prompts.yml` | 提示词文件路径 |
| `data/external/records.csv` | 模拟用户使用记录 |

请始终在项目根目录启动命令，否则相对路径 `chroma_db`、`data` 可能找不到。

## 工具说明

部分工具为演示用 mock（天气、位置、用户 ID、当前月份），非正式线上接口。`rag_summarize` 与 `fetch_external_data` 会读取本地知识库和 CSV。

## 注意事项

- **不要提交** `DASHSCOPE_API_KEY`、`.env`、本地日志和向量库。仓库已提供 `.gitignore`。
- 知识库文件只扫描 `data/` 下一层的 `txt` / `pdf`。
- 更换知识库文档后请重新执行 `python -m rag.vector_store`。

## License

本项目仅供学习与演示使用。
