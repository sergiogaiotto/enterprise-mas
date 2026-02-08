# Enterprise MAS — Multi-Agent System com LangGraph

Sistema multi-agente empresarial.

Implementação completa do padrão **Plan → Retrieve → Execute → Review → Respond** usando LangGraph como orquestrador de DAG.

---

## Arquitetura

```
                    ┌──────────┐
                    │ Planning │
                    │  Agent   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
              ┌────►│  Search  │◄────┐
              │     │  Agent   │     │
              │     └────┬─────┘     │
              │          │           │
              │     ┌────▼─────┐     │
              │     │ Executor │     │
              │     │  Agent   │─────┘  (loop: mais sub-tasks)
              │     └────┬─────┘
              │          │
              │     ┌────▼─────┐
              │     │ Responder│
              │     │  Agent   │◄────┐
              │     └────┬─────┘     │
              │          │           │  (revisão)
              │     ┌────▼─────┐     │
              │     │  Review  │─────┘
              │     │  Agent   │
              │     └────┬─────┘
              │          │
              │     ┌────▼─────┐
              │     │ Finalize │
              │     └──────────┘
```

## Agentes

| Agente | Função |
|--------|--------|
| **Planning** | Decompõe a query em sub-tarefas ordenadas (1-5) |
| **Search** | Multi-hop retrieval: ChromaDB local + DuckDuckGo web search + grading + rephrase |
| **Executor** | Executa cada sub-tarefa com contexto recuperado |
| **Responder** | Sintetiza resultados em resposta coerente |
| **Review** | QA com critérios de acurácia, completude e coerência |

## Stack

- **Backend**: Python 3.11+, FastAPI, LangGraph, LangChain
- **LLM**: OpenAI (gpt-4.1-mini default, configurável)
- **Vector Store**: ChromaDB com OpenAI Embeddings
- **Web Search**: DuckDuckGo (sem API key)
- **Frontend**: HTML5 + Tailwind CSS (responsive)

## Setup

```bash
# 1. Clone e entre no diretório
cd enterprise-mas

# 2. Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Configure a API key
cp .env.example .env
# Edite .env e insira sua OPENAI_API_KEY

# 5. Execute
uvicorn main:app --reload
```

Acesse `http://localhost:8000` no navegador.

## API Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/api/chat` | Executa query no pipeline multi-agente |
| `POST` | `/api/ingest` | Adiciona documentos ao ChromaDB |
| `GET` | `/api/health` | Health check |

### Exemplo — Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explique LangGraph para enterprise"}'
```

### Exemplo — Ingestão

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Documento sobre políticas internas...", "Manual de operações..."]}'
```

## Estrutura

```
enterprise-mas/
├── main.py                    # Entry point FastAPI
├── requirements.txt
├── .env
├── templates/
│   └── default.html           # Frontend (Tailwind CSS)
├── static/                    # Assets estáticos
└── app/
    ├── core/
    │   ├── config.py          # Configuração (.env)
    │   └── state.py           # AgentState (TypedDict)
    ├── agents/
    │   ├── planner.py         # Planning Agent
    │   ├── searcher.py        # Search Agent (multi-hop)
    │   ├── executor.py        # Executor Agent
    │   ├── reviewer.py        # Review Agent
    │   ├── responder.py       # Response Generation Agent
    │   └── router.py          # Conditional edge logic
    ├── tools/
    │   ├── web_search.py      # DuckDuckGo search
    │   └── knowledge_base.py  # ChromaDB vector store
    ├── api/
    │   └── routes.py          # FastAPI routes
    └── graph.py               # LangGraph DAG compilation
```

