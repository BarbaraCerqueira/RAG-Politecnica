# PoliGPT

Assistente virtual baseado em RAG (*Retrieval-Augmented Generation*) para alunos, professores e funcionários da Escola Politécnica da UFRJ. As respostas são geradas a partir do conteúdo público do site [poli.ufrj.br](https://poli.ufrj.br), coletado e processado por um pipeline de dados na Azure.

Projeto de Graduação em Engenharia de Computação e Informação — UFRJ.

## Arquitetura

```
poli.ufrj.br
  │  (web scraping semanal)
  ▼
Databricks ── control → raw → trusted → refined ──► Azure SQL
                                                      │  (indexador com change tracking)
                                                      ▼
                                   Azure AI Search: chunking + embeddings ──► índice vetorial
                                                                               │
Frontend React (Static Web App) ──► Backend (Azure Function + LangGraph + OpenAI) ◄─┘
```

## Estrutura do repositório

| Pasta | Conteúdo |
|---|---|
| [`infra/`](infra/) | Infraestrutura como código (Terraform) para Azure e Databricks |
| [`pipeline/`](pipeline/) | Notebooks do pipeline de dados no Databricks, organizados por camada |
| [`search/`](search/) | Scripts de criação dos objetos do Azure AI Search (`setup/`) e função de embeddings (`embedding-function/`) usada por ele |
| [`backend/`](backend/) | API do chat (Azure Function com agente RAG em LangGraph) |
| [`frontend/`](frontend/) | Interface web do chat (React + Vite + Tailwind) |
| [`experiments/`](experiments/) | Notebooks e scripts exploratórios usados durante o desenvolvimento |

### Pipeline de dados

| Camada | Notebook | Descrição |
|---|---|---|
| Control | `01_control/sitemap_extraction` | Lê o sitemap do site e registra URLs e datas de modificação |
| Raw | `02_raw/raw_data_ingestion` | Baixa HTMLs e PDFs novos ou alterados para o Data Lake |
| Trusted | `03_trusted/trusted_poli_consolidated_{html,pdf}` | Extrai texto e metadados em tabelas Delta (PDFs escaneados passam por OCR com Tesseract) |
| Refined | `04_refined/tb_poli_knowledge_base` | Filtra, unifica e grava a base de conhecimento no Azure SQL |
| Indexing | `05_indexing/run_search_indexer` | Dispara o indexador do AI Search |

## Deploy

| Componente | Como publicar |
|---|---|
| Infraestrutura (Azure + Databricks) | `terraform apply` em [`infra/`](infra/) — ver [`infra/README.md`](infra/README.md) |
| Function do chat | `func azure functionapp publish func-poligpt-chat-dev --python` a partir de `backend/` |
| Function de embeddings | `func azure functionapp publish func-poligpt-embed-dev --python` a partir de `search/embedding-function/` |
| Objetos do AI Search | scripts de [`search/setup/`](search/setup/) — ver [`search/README.md`](search/README.md) |
| Pipeline de dados | O job `poligpt_workflow` busca os notebooks direto do GitHub (branch `main`) a cada execução, então basta fazer merge na `main`. |
