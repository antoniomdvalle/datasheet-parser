import threading
import tkinter as tk
from tkinter import scrolledtext
from pathlib import Path
from llama_index.core import Settings, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ==========================================
# 1. CONFIGURAÇÕES OTIMIZADAS PARA VELOCIDADE
# ==========================================
OLLAMA_URL = "http://localhost:11434"
PERSIST_DIR = "./storage"

# LLM leve (qwen2.5:1.5b ~980MB) com limite de 150 tokens na resposta
Settings.llm = Ollama(
    model="qwen2.5:1.5b",
    base_url=OLLAMA_URL,
    temperature=0.0,
    request_timeout=60.0,
    context_window=4096,
    additional_kwargs={"num_predict": 150}
)

# Embedding ultrarrápido (45MB)
Settings.embed_model = OllamaEmbedding(
    model_name="all-minilm",
    base_url=OLLAMA_URL
)

# System Prompt rígido para respostas diretas em 1 ou 2 frases
QA_PROMPT_TMPL = (
    "You are a fast, objective technical assistant for laser sensors. "
    "Answer the question objectively in 1 or 2 sentences using ONLY the provided manual context below. "
    "Do not add extra explanations or filler words. If the answer is not in the context, say 'Information not found in the manual.'\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

# ==========================================
# 2. INTERFACE GRÁFICA (TKINTER COM STREAMING)
# ==========================================
class SensorSupportApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laser Sensor Support - Fast AI (1.5B)")
        self.root.geometry("680x540")
        self.root.configure(bg="#F4F7FA")
        self.root.resizable(True, True)

        self.query_engine = None
        self.criar_widgets()

        threading.Thread(target=self.carregar_indice, daemon=True).start()

    def criar_widgets(self):
        # Cabeçalho Azul
        header = tk.Frame(self.root, bg="#1E56A0", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        lbl_titulo = tk.Label(
            header,
            text="Laser Sensor Support - Fast Search",
            font=("Segoe UI", 14, "bold"),
            bg="#1E56A0",
            fg="#FFFFFF"
        )
        lbl_titulo.pack(pady=15, padx=20, anchor="w")

        # Área Principal
        container = tk.Frame(self.root, bg="#F4F7FA", padx=20, pady=15)
        container.pack(fill=tk.BOTH, expand=True)

        lbl_pergunta = tk.Label(
            container,
            text="Sua pergunta (em inglês):",
            font=("Segoe UI", 10, "bold"),
            bg="#F4F7FA",
            fg="#163172"
        )
        lbl_pergunta.pack(anchor="w", pady=(0, 5))

        frame_input = tk.Frame(container, bg="#F4F7FA")
        frame_input.pack(fill=tk.X, pady=(0, 15))

        self.entry_prompt = tk.Entry(
            frame_input,
            font=("Segoe UI", 11),
            bg="#FFFFFF",
            fg="#1A1A1A",
            relief=tk.SOLID,
            bd=1
        )
        self.entry_prompt.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 10))
        self.entry_prompt.bind("<Return>", lambda event: self.enviar_pergunta())

        self.btn_enviar = tk.Button(
            frame_input,
            text="Perguntar",
            font=("Segoe UI", 10, "bold"),
            bg="#1E56A0",
            fg="#FFFFFF",
            activebackground="#163172",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            padx=20,
            pady=6,
            state=tk.DISABLED,
            command=self.enviar_pergunta
        )
        self.btn_enviar.pack(side=tk.RIGHT)

        lbl_resposta = tk.Label(
            container,
            text="Resposta:",
            font=("Segoe UI", 10, "bold"),
            bg="#F4F7FA",
            fg="#163172"
        )
        lbl_resposta.pack(anchor="w", pady=(0, 5))

        self.txt_resposta = scrolledtext.ScrolledText(
            container,
            font=("Segoe UI", 11),
            bg="#FFFFFF",
            fg="#1A1A1A",
            relief=tk.SOLID,
            bd=1,
            wrap=tk.WORD,
            height=12,
            state=tk.DISABLED
        )
        self.txt_resposta.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        frame_footer = tk.Frame(container, bg="#F4F7FA")
        frame_footer.pack(fill=tk.X)

        self.lbl_status = tk.Label(
            frame_footer,
            text="Carregando índice vetorial...",
            font=("Segoe UI", 9),
            bg="#F4F7FA",
            fg="#556B82"
        )
        self.lbl_status.pack(side=tk.LEFT)

        self.btn_copiar = tk.Button(
            frame_footer,
            text="Copiar Resposta",
            font=("Segoe UI", 9, "bold"),
            bg="#D6E4F0",
            fg="#163172",
            activebackground="#B8D0E8",
            activeforeground="#163172",
            relief=tk.FLAT,
            cursor="hand2",
            padx=14,
            pady=5,
            command=self.copiar_resposta
        )
        self.btn_copiar.pack(side=tk.RIGHT)

    def carregar_indice(self):
        try:
            if not Path(PERSIST_DIR).exists():
                self.atualizar_status("Erro: pasta ./storage não encontrada. Rode 2_criar_indice.py.")
                return

            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)

            # top_k reduzido para 3 e streaming ativado
            self.query_engine = index.as_query_engine(
                similarity_top_k=3,
                text_qa_template=QA_PROMPT,
                streaming=True
            )
            self.root.after(0, self.ao_carregar_indice)
        except Exception as e:
            self.atualizar_status(f"Erro ao carregar índice: {e}")

    def ao_carregar_indice(self):
        self.btn_enviar.config(state=tk.NORMAL)
        self.lbl_status.config(text="Pronto (Qwen 1.5B + MiniLM).")
        self.entry_prompt.focus()

    def enviar_pergunta(self):
        pergunta = self.entry_prompt.get().strip()
        if not pergunta or not self.query_engine:
            return

        self.btn_enviar.config(state=tk.DISABLED)
        self.lbl_status.config(text="Gerando resposta...")
        self.limpar_texto_resposta()

        threading.Thread(target=self.processar_consulta_stream, args=(pergunta,), daemon=True).start()

    def processar_consulta_stream(self, pergunta):
        try:
            streaming_response = self.query_engine.query(pergunta)
            for token in streaming_response.response_gen:
                self.root.after(0, lambda t=token: self.adicionar_token_resposta(t))

            self.root.after(0, self.finalizar_consulta)
        except Exception as e:
            self.root.after(0, lambda: self.exibir_erro(f"Erro: {e}"))

    def limpar_texto_resposta(self):
        self.txt_resposta.config(state=tk.NORMAL)
        self.txt_resposta.delete("1.0", tk.END)
        self.txt_resposta.config(state=tk.DISABLED)

    def adicionar_token_resposta(self, token):
        self.txt_resposta.config(state=tk.NORMAL)
        self.txt_resposta.insert(tk.END, token)
        self.txt_resposta.see(tk.END)
        self.txt_resposta.config(state=tk.DISABLED)

    def finalizar_consulta(self):
        self.lbl_status.config(text="Consulta concluída.")
        self.btn_enviar.config(state=tk.NORMAL)

    def exibir_erro(self, msg):
        self.limpar_texto_resposta()
        self.adicionar_token_resposta(msg)
        self.lbl_status.config(text="Falha na consulta.")
        self.btn_enviar.config(state=tk.NORMAL)

    def atualizar_status(self, msg):
        self.root.after(0, lambda: self.lbl_status.config(text=msg))

    def copiar_resposta(self):
        texto = self.txt_resposta.get("1.0", tk.END).strip()
        if not texto:
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(texto)
        self.root.update()

        self.btn_copiar.config(text="Copiado!", bg="#2E8B57", fg="#FFFFFF")
        self.root.after(1500, lambda: self.btn_copiar.config(
            text="Copiar Resposta", bg="#D6E4F0", fg="#163172"
        ))

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorSupportApp(root)
    root.mainloop()