import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
from pathlib import Path
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ==========================================
# 1. CONFIGURAÇÕES DO RAG / OLLAMA
# ==========================================
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

# ==========================================
# 2. INTERFACE GRÁFICA (TKINTER)
# ==========================================
class SensorSupportApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laser Sensor Support - AI")
        self.root.geometry("680x540")
        self.root.configure(bg="#F4F7FA")  # Fundo claro suave
        self.root.resizable(True, True)

        self.query_engine = None
        self.criar_widgets()

        # Carrega o índice em segundo plano para a janela abrir na hora
        threading.Thread(target=self.carregar_indice, daemon=True).start()

    def criar_widgets(self):
        # --- Cabeçalho Azul ---
        header = tk.Frame(self.root, bg="#1E56A0", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        lbl_titulo = tk.Label(
            header,
            text="Laser Sensor Support - AI",
            font=("Segoe UI", 14, "bold"),
            bg="#1E56A0",
            fg="#FFFFFF"
        )
        lbl_titulo.pack(pady=15, padx=20, anchor="w")

        # --- Área Principal ---
        container = tk.Frame(self.root, bg="#F4F7FA", padx=20, pady=15)
        container.pack(fill=tk.BOTH, expand=True)

        # Label do Prompt
        lbl_pergunta = tk.Label(
            container,
            text="Sua pergunta (em inglês):",
            font=("Segoe UI", 10, "bold"),
            bg="#F4F7FA",
            fg="#163172"
        )
        lbl_pergunta.pack(anchor="w", pady=(0, 5))

        # Caixa de Entrada do Prompt + Botão Enviar
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
            state=tk.DISABLED,  # Habilita assim que o índice terminar de carregar
            command=self.enviar_pergunta
        )
        self.btn_enviar.pack(side=tk.RIGHT)

        # Label da Resposta
        lbl_resposta = tk.Label(
            container,
            text="Resposta:",
            font=("Segoe UI", 10, "bold"),
            bg="#F4F7FA",
            fg="#163172"
        )
        lbl_resposta.pack(anchor="w", pady=(0, 5))

        # Caixa de Texto da Resposta
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

        # --- Rodapé com Status e Botão Copiar ---
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
                self.atualizar_status("Erro: pasta ./storage não encontrada.")
                return

            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            self.query_engine = index.as_query_engine(
                similarity_top_k=4,
                response_mode="compact"
            )
            self.root.after(0, self.ao_carregar_indice)
        except Exception as e:
            self.atualizar_status(f"Erro ao carregar índice: {e}")

    def ao_carregar_indice(self):
        self.btn_enviar.config(state=tk.NORMAL)
        self.lbl_status.config(text="Pronto para consultas.")
        self.entry_prompt.focus()

    def enviar_pergunta(self):
        pergunta = self.entry_prompt.get().strip()
        if not pergunta or not self.query_engine:
            return

        # Bloqueia o botão enquanto o Ollama gera a resposta
        self.btn_enviar.config(state=tk.DISABLED)
        self.lbl_status.config(text="Consultando o manual...")
        self.definir_texto_resposta("Processando sua pergunta, aguarde...")

        # Roda a consulta em uma Thread separada para a tela não travar ("Não respondendo")
        threading.Thread(target=self.processar_consulta, args=(pergunta,), daemon=True).start()

    def processar_consulta(self, pergunta):
        try:
            response = self.query_engine.query(pergunta)
            texto_final = str(response).strip()
            self.root.after(0, lambda: self.exibir_resultado(texto_final, "Consulta concluída."))
        except Exception as e:
            self.root.after(0, lambda: self.exibir_resultado(f"Erro: {e}", "Falha na consulta."))

    def exibir_resultado(self, texto, status):
        self.definir_texto_resposta(texto)
        self.lbl_status.config(text=status)
        self.btn_enviar.config(state=tk.NORMAL)

    def definir_texto_resposta(self, texto):
        self.txt_resposta.config(state=tk.NORMAL)
        self.txt_resposta.delete("1.0", tk.END)
        self.txt_resposta.insert(tk.END, texto)
        self.txt_resposta.config(state=tk.DISABLED)

    def atualizar_status(self, msg):
        self.root.after(0, lambda: self.lbl_status.config(text=msg))

    def copiar_resposta(self):
        texto = self.txt_resposta.get("1.0", tk.END).strip()
        if not texto:
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(texto)
        self.root.update()  # Mantém na área de transferência do Windows

        # Feedback visual rápido no botão
        self.btn_copiar.config(text="Copiado!", bg="#2E8B57", fg="#FFFFFF")
        self.root.after(1500, lambda: self.btn_copiar.config(
            text="Copiar Resposta", bg="#D6E4F0", fg="#163172"
        ))

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorSupportApp(root)
    root.mainloop()