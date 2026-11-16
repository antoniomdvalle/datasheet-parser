from pathlib import Path
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

OLLAMA_URL = "http://localhost:11434"
PERSIST_DIR = "./storage"

Settings.llm = Ollama(
    model="qwen2.5-coder:7b",
    base_url=OLLAMA_URL,
    temperature=0.1,
    request_timeout=180.0,
    context_window=32768
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_URL
)

if not Path(PERSIST_DIR).exists():
    raise FileNotFoundError("Índice não encontrado. Rode o 2_criar_indice.py primeiro.")

print("Carregando índice...")
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(
    similarity_top_k=4,
    response_mode="compact"
)

print("\n================================")
print("   LASER SENSOR SUPPORT - AI")
print("================================")
print("Type 'quit' to end this conversation.\n")

while True:
    question = input("You: ")

    if question.lower().strip() in ["exit", "quit"]:
        print("Finishing session...")
        break

    if not question.strip():
        continue

    try:
        # Chamada única (no seu código original estava chamando duas vezes)
        response = query_engine.query(question)

        print("\nAgent:", response)

        print("\n===== RETRIEVED CONTEXT =====")
        for i, source in enumerate(response.source_nodes, 1):
            arquivo = source.metadata.get("file_name", "Desconhecido")
            score = f"{source.score:.3f}" if source.score is not None else "N/A"
            print(f"\n--- CHUNK {i} | Arquivo: {arquivo} | Relevância: {score} ---")
            print(source.node.get_content())

        print("\n" + "=" * 45 + "\n")

    except Exception as error:
        print(f"\nError during query: {error}\n")