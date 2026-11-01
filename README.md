# Forward Deployed Engineer: GenAI Notes

My study notes for becoming a Forward Deployed Engineer (FDE) who builds with generative AI. The notes go from "what is an FDE?" to LLMs, RAG, agents, MCP, evals, security and agent harnesses.

## How to use these notes

Every note has the same shape:

1. **In plain words** - a short summary in simple English.
2. The main ideas, with tables and diagrams.
3. **Picture it** - extra visuals (diagrams and examples) to make the idea stick.
4. **Key points to remember** and **Interview angles**.
5. **Further reading** - original papers, official docs and guides for deeper study.

Diagrams use [Mermaid](https://mermaid.js.org/), which GitHub shows automatically.

## Contents

| # | Note |
|---|---|
| 1 | [Forward Deployed Engineering: The Role and the Rise of LLMs](notes/01-forward-deployed-engineering.md) |
| 2 | [How LLMs Actually Work: Tokens, Attention and Transformers](notes/02-how-llms-work.md) |
| 3 | [Integrating an LLM with Your Application](notes/03-integrating-llm-with-your-app.md) |
| 4 | [Build Your First AI Chatbot: Context Window, Memory and Hallucinations](notes/04-first-ai-chatbot.md) |
| 5 | [Introduction to AI Agents: Build Your First Agent](notes/05-ai-agents.md) |
| 6 | [Build ChatGPT-Style Streaming: How AI Streaming Works](notes/06-streaming.md) |
| 7 | [Vectors and Embeddings: How AI Understands Meaning](notes/07-vectors-and-embeddings.md) |
| 8 | [Build a Netflix-Style Recommendation System with Vector Embeddings](notes/08-recommendation-with-embeddings.md) |
| 9 | [Why We Need Vector Databases: KNN and Vector Search](notes/09-why-vector-databases.md) |
| 10 | [How Vector Databases Actually Work: ANN, HNSW, IVF and Product Quantization](notes/10-how-vector-databases-work.md) |
| 11 | [RAG Explained with a Real Project: AI Customer Support Chatbot](notes/11-rag.md) |
| 12 | [Advanced RAG: Chunking, Hybrid Search, Reranking and More](notes/12-advanced-rag.md) |
| 13 | [Structured Outputs in LLMs: Responses Applications Can Use](notes/13-structured-outputs.md) |
| 14 | [MCP Server Explained: Model Context Protocol for AI Agents](notes/14-mcp-server.md) |
| 15 | [MCP Resources and Prompts Explained: Beyond MCP Tools](notes/15-mcp-resources-prompts.md) |
| 16 | [How AI Understands Images: Build a Multimodal AI App](notes/16-multimodal-ai.md) |
| 17 | [How to Test AI Applications: GenAI Evals, LLM-as-a-Judge and RAG Testing](notes/17-testing-ai-apps-and-evals.md) |
| 18 | [Building and Testing an E-Commerce AI Chatbot: RAG, Tools and AI Evals](notes/18-ecommerce-chatbot.md) |
| 19 | [AI Security and Guardrails: Prompt Injection, Jailbreaks and Safe Tool Use](notes/19-ai-security-guardrails.md) |
| 20 | [Agent Harness: Building Reliable AI Agents](notes/20-agent-harness.md) |

## Suggested study path

```mermaid
flowchart LR
  A["1. The role"] --> B["2-4. LLM basics<br/>and first chatbot"]
  B --> C["5-6. Agents<br/>and streaming"]
  C --> D["7-12. Embeddings,<br/>vector DBs, RAG"]
  D --> E["13-15. Structured output<br/>and MCP"]
  E --> F["16. Multimodal"]
  F --> G["17-19. Evals<br/>and security"]
  G --> H["20. Agent harness"]
```

## Notes on accuracy

The diagrams, numbers in examples and reading lists were added to help study. The examples (such as probabilities or vector sizes) are for teaching and are not benchmarks. Check the linked sources for exact details.
