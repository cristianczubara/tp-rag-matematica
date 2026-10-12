"""
RAG sobre una CARPETA DE PDFs (Ollama + ChromaDB persistente) - v2.

Cambios respecto de la version original, pensados para PDFs de matematica
generados con LaTeX (formulas, simbolos, "cajas" de definicion/ejemplo):

1. Extraccion con PyMuPDF (fitz) en vez de pypdf: pypdf pegaba palabras sin
   espacio (ej: "denominatangente" en vez de "denomina tangente"), lo cual
   rompia justo los terminos clave que se buscan en las consultas.
2. Correccion de acentos: la fuente del PDF codifica las vocales acentuadas
   como "acento suelto + letra" (ej. "´on" en vez de "ón"). Se corrige con
   un mapeo de reemplazos antes de indexar.
3. Chunking sobre el TEXTO COMPLETO del PDF (no pagina por pagina): antes,
   un ejemplo que cruzaba de una pagina a la siguiente (como el de "recta
   tangente a una circunferencia", que sigue de la pag. 45 a la 46) quedaba
   partido en dos chunks que el retrieval podia no recuperar juntos.
4. Chunks mas grandes con mas overlap, para no separar el enunciado de un
   ejemplo de su formula/resultado.
5. TOP_K mas alto, ya que los chunks ahora tienen mas contexto matematico.

Requisitos:
    - Ollama corriendo con:  ollama pull gemma3 ; ollama pull nomic-embed-text
    - pip install pymupdf chromadb ollama --break-system-packages

Uso: igual que la version original.
    python rag_pdf_entrega_tp.py "¿Qué es una recta tangente a una circunferencia?"
"""

from __future__ import annotations

import os
import re
import sys

import chromadb
import ollama
import pymupdf as fitz

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_pdf_v2")
COLLECTION_NAME = "pdf_docs_v2"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3"
TOP_K = 10
CHUNK_SIZE = 1500     # caracteres por chunk (mas grande: no cortar ejemplos con formulas)
CHUNK_OVERLAP = 300   # mas overlap para no perder el hilo entre chunk y chunk

# Mapeo para arreglar acentos mal codificados por la fuente del PDF
# (el glifo de acento aparece SUELTO antes de la vocal, en vez de combinado)
ACCENT_FIX = {
    "´a": "á", "´e": "é", "´ı": "í", "´o": "ó", "´u": "ú",
    "´A": "Á", "´E": "É", "´I": "Í", "´O": "Ó", "´U": "Ú",
    "¨u": "ü", "˜n": "ñ",
}


def fix_accents(texto: str) -> str:
    """Corrige vocales acentuadas mal codificadas por la fuente del PDF."""
    for roto, bien in ACCENT_FIX.items():
        texto = texto.replace(roto, bien)
    return texto


def embed(texto: str) -> list[float]:
    """Devuelve el embedding de un texto usando Ollama."""
    return ollama.embeddings(model=EMBED_MODEL, prompt=texto)["embedding"]


def chunk_text_with_offsets(texto: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Parte un texto en fragmentos, devolviendo tambien el offset de caracter
    donde empieza cada chunk (para poder mapear despues a un numero de pagina)."""
    if not texto:
        return []
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + size
        chunks.append((inicio, texto[inicio:fin]))
        inicio += size - overlap
    return chunks


def extraer_texto_completo(ruta: str):
    """Extrae el texto de TODO el PDF con PyMuPDF, concatenando paginas,
    y arma un mapa offset_de_caracter -> numero_de_pagina para poder
    citar la pagina correcta despues de chunkear el documento completo."""
    doc = fitz.open(ruta)
    partes = []
    mapa_paginas = []  # lista de (offset_inicial, numero_pagina)
    offset = 0
    for n_pagina, page in enumerate(doc, start=1):
        texto = page.get_text() or ""
        texto = fix_accents(texto)
        texto = " ".join(texto.split())  # normaliza espacios/saltos de linea
        if not texto:
            continue
        mapa_paginas.append((offset, n_pagina))
        partes.append(texto)
        offset += len(texto) + 1  # +1 por el separador que agregamos abajo
    doc.close()
    texto_completo = " ".join(partes)
    return texto_completo, mapa_paginas


def pagina_para_offset(offset: int, mapa_paginas: list[tuple[int, int]]) -> int:
    """Dado un offset de caracter en el texto completo, devuelve la pagina
    correspondiente (la ultima entrada del mapa cuyo offset es <= al dado)."""
    pagina = mapa_paginas[0][1] if mapa_paginas else 1
    for offset_inicio, n_pagina in mapa_paginas:
        if offset_inicio <= offset:
            pagina = n_pagina
        else:
            break
    return pagina


def extraer_chunks_pdf(ruta: str) -> list[dict]:
    """Extrae el texto de un PDF y devuelve una lista de chunks con metadata,
    chunkeando sobre el documento completo (no pagina por pagina)."""
    nombre = os.path.basename(ruta)
    texto_completo, mapa_paginas = extraer_texto_completo(ruta)
    items = []
    for i, (offset, ch) in enumerate(chunk_text_with_offsets(texto_completo)):
        pagina = pagina_para_offset(offset, mapa_paginas)
        items.append({
            "id": f"{nombre}::c{i}",
            "text": ch,
            "metadata": {"source": nombre, "page": pagina},
        })
    return items


def get_collection(reindex: bool = False):
    """Abre (o crea) la coleccion persistente de Chroma. Con reindex=True la borra antes."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    if reindex:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION_NAME)


def pdf_ya_indexado(collection, nombre: str) -> bool:
    res = collection.get(where={"source": nombre}, limit=1)
    return len(res.get("ids", [])) > 0


def indexar(collection, docs_dir: str) -> None:
    pdfs = [f for f in sorted(os.listdir(docs_dir)) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"[aviso] No hay PDFs en {docs_dir}. Poné tus archivos ahi y volvé a correr.")
        return

    total_nuevos = 0
    for nombre in pdfs:
        if pdf_ya_indexado(collection, nombre):
            print(f"  = {nombre} (ya indexado, se saltea)")
            continue
        ruta = os.path.join(docs_dir, nombre)
        try:
            items = extraer_chunks_pdf(ruta)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {nombre}: no se pudo leer ({e})")
            continue
        if not items:
            print(f"  ! {nombre}: sin texto extraible (¿PDF escaneado?)")
            continue
        collection.add(
            ids=[it["id"] for it in items],
            embeddings=[embed(it["text"]) for it in items],
            documents=[it["text"] for it in items],
            metadatas=[it["metadata"] for it in items],
        )
        total_nuevos += len(items)
        print(f"  + {nombre}: {len(items)} chunks indexados")

    print(f"\nIndexado listo. Chunks nuevos: {total_nuevos}. "
          f"Total en la base: {collection.count()}.\n")


def recuperar(collection, pregunta: str, k: int = TOP_K) -> list[dict]:
    q = embed(pregunta)
    res = collection.query(query_embeddings=[q], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [{"text": d, "meta": m} for d, m in zip(docs, metas)]


def generar_respuesta(pregunta: str, contexto: list[dict]) -> str:
    bloques = []
    for c in contexto:
        fuente = f"{c['meta'].get('source')} (pag. {c['meta'].get('page')})"
        bloques.append(f"[{fuente}]\n{c['text']}")
    contexto_str = "\n\n".join(bloques)
    prompt = f"""Respondé la pregunta usando SOLO la informacion del contexto extraido
de los PDFs. Si el contexto no alcanza, decilo. Citá las fuentes entre corchetes.

Contexto:
{contexto_str}

Pregunta: {pregunta}

Respuesta:"""
    resp = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


def consultar(collection, pregunta: str) -> None:
    print(f"\n=== Pregunta: {pregunta} ===")
    contexto = recuperar(collection, pregunta)
    if not contexto:
        print("No hay documentos indexados todavia.")
        return
    print("\n--- Fragmentos recuperados ---")
    for c in contexto:
        fuente = f"{c['meta'].get('source')} p.{c['meta'].get('page')}"
        print(f"  * [{fuente}] {c['text'][:80]}...")
    print("\n--- Respuesta del modelo ---")
    print(generar_respuesta(pregunta, contexto))


def main() -> None:
    argv = sys.argv[1:]
    reindex = "--reindex" in argv
    argv = [a for a in argv if a != "--reindex"]

    docs_dir = DOCS_DIR
    if "--docs" in argv:
        i = argv.index("--docs")
        docs_dir = argv[i + 1]
        del argv[i:i + 2]

    os.makedirs(docs_dir, exist_ok=True)
    collection = get_collection(reindex=reindex)

    print(f"Carpeta de PDFs: {docs_dir}")
    print("Indexando PDFs (incremental)...")
    indexar(collection, docs_dir)

    if collection.count() == 0:
        print("No hay nada indexado. Agregá PDFs y volvé a correr.")
        return

    if argv:
        consultar(collection, " ".join(argv))
    else:
        print("Escribí una pregunta (o 'salir' para terminar):")
        while True:
            try:
                pregunta = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if pregunta.lower() in {"salir", "exit", "quit"}:
                break
            if pregunta:
                consultar(collection, pregunta)


if __name__ == "__main__":
    main()
