import shutil
from pathlib import Path
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding

OLLAMA_URL = "http://localhost:11434"
PASTA_TEXTO = "./texto_extraido"
PERSIST_DIR = "./storage"

# Embedding leve de 45 MB
Settings.embed_model = OllamaEmbedding(
    model_name="all-minilm",
    base_url=OLLAMA_URL
)

Settings.chunk_size = 512
Settings.chunk_overlap = 50

print("Carregando textos extraídos...")
documents = SimpleDirectoryReader(input_dir=PASTA_TEXTO).load_data()

if not documents:
    raise ValueError("Nenhum arquivo .txt encontrado em ./texto_extraido.")

if Path(PERSIST_DIR).exists():
    shutil.rmtree(PERSIST_DIR)

print("Criando novo índice vetorial com all-minilm...")
index = VectorStoreIndex.from_documents(documents, show_progress=True)
index.storage_context.persist(persist_dir=PERSIST_DIR)
print("Índice salvo com sucesso em ./storage!")