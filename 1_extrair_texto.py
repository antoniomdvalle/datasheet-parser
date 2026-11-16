import re
from pathlib import Path
from pypdf import PdfReader

PASTA_PDF = Path("./dados_manual")
PASTA_SAIDA = Path("./texto_extraido")
PASTA_SAIDA.mkdir(exist_ok=True)

pdfs = list(PASTA_PDF.glob("*.pdf"))

if not pdfs:
    print("Nenhum PDF encontrado em ./dados_manual.")
    exit()

for pdf_path in pdfs:
    print(f"Extraindo: {pdf_path.name}...")
    reader = PdfReader(str(pdf_path))
    paginas_limpas = []

    for i, page in enumerate(reader.pages, 1):
        texto = page.extract_text() or ""
        # Remove múltiplos espaços e quebras de linha excessivas
        texto = re.sub(r'[ \t]+', ' ', texto)
        texto = re.sub(r'\n{3,}', '\n\n', texto).strip()

        if texto:
            paginas_limpas.append(f"--- PÁGINA {i} ---\n{texto}")

    arquivo_txt = PASTA_SAIDA / f"{pdf_path.stem}.txt"
    arquivo_txt.write_text("\n\n".join(paginas_limpas), encoding="utf-8")
    print(f"Salvo em: {arquivo_txt} ({len(reader.pages)} páginas)")