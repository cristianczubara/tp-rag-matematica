# RAG sobre los apuntes del Trayecto Matemática (ITBA)

Trabajo práctico *Aplicaciones con Modelos de Lenguaje*, **Opción A: RAG**.

El sistema responde preguntas sobre los apuntes teóricos del Trayecto Matemática de la Secretaría de Admisión del ITBA. Antes de generar cada respuesta, recupera los fragmentos relevantes de los PDFs. Todo corre en local con modelos abiertos (Ollama), sin APIs pagas.

---

## 1. Fundamentos teóricos

### El problema que resuelve RAG

Un LLM responde a partir de lo que aprendió durante el entrenamiento, su **conocimiento paramétrico**. Ese conocimiento tiene tres limitaciones relevantes para este caso:

- **No conoce documentos privados o específicos.** Puede explicar qué es un número complejo, pero no sabe cómo lo presentan *estos* apuntes: qué notación usan, qué analogías, qué convenciones (por ejemplo, la "condición inicial" que se pide en cada ejercicio).
- **Alucina.** Cuando no sabe algo, suele generar una respuesta plausible en lugar de abstenerse.
- **No cita fuentes.** No hay forma de verificar de dónde salió una afirmación.

**RAG** (*Retrieval-Augmented Generation*, Lewis et al., 2020) separa *saber dónde buscar* de *saber redactar*. Ante cada pregunta, primero se **recuperan** los fragmentos más relevantes de una colección de documentos. Después se le pide al LLM que **genere** la respuesta usando solo esos fragmentos. El modelo pasa de "recordar" a "leer y resumir", una tarea en la que es mucho más confiable.

### Componentes

| Concepto | Qué es | En este proyecto |
|---|---|---|
| **Chunking** | Dividir cada documento en fragmentos del tamaño adecuado para recuperar y para entrar en el prompt | Ventana de 1500 caracteres con 300 de solapamiento, sobre el texto completo de cada PDF |
| **Embedding** | Vector numérico que representa el significado de un texto. Textos de significado parecido quedan cerca en el espacio vectorial | `nomic-embed-text` (768 dimensiones) |
| **Base vectorial** | Almacena los vectores y permite buscar los más cercanos a una consulta de forma eficiente (índice HNSW) | ChromaDB persistente, distancia coseno |
| **Top-k** | Cantidad de fragmentos recuperados por consulta. Pocos: puede faltar información. Muchos: se suma ruido y se alarga el prompt | k = 10 |
| **Generación aumentada** | El prompt incluye los fragmentos y obliga al modelo a basarse en ellos, decir si no alcanzan y citar la fuente | `gemma3`, temperatura 0 |

La similitud coseno entre la consulta *q* y un fragmento *d* es `cos(q, d) = (q · d) / (‖q‖ ‖d‖)`. Chroma devuelve la distancia `1 − cos`, por lo que menor distancia significa mayor parecido.

---

## 2. Arquitectura

```
                 INDEXACIÓN (una vez)                          CONSULTA (cada pregunta)

 docs/*.pdf                                                   Pregunta
     │                                                            │
     ▼                                                            ▼
 PyMuPDF: extracción ─► corrección de acentos              nomic-embed-text
     │                                                            │
     ▼                                                            ▼
 Texto completo + mapa offset→página                   ┌──► ChromaDB: top-k chunks
     │                                                 │          │
     ▼                                                 │          ▼
 Chunks de 1500 car. (solapamiento 300)                │   Prompt = instrucciones
     │                                                 │          + chunks con [archivo, pág.]
     ▼                                                 │          + pregunta
 nomic-embed-text ─► ChromaDB (persistente) ───────────┘          │
                                                                  ▼
                                                     gemma3 ─► respuesta con citas
```

### Decisiones de diseño y por qué

1. **PyMuPDF en lugar de pypdf.** Con `pypdf`, la extracción pegaba palabras (*denominatangente*) y rompía justo los términos clave de las consultas.
2. **Corrección de acentos.** La fuente de LaTeX codifica "ó" como acento suelto seguido de la vocal (`Definici´on`). Sin corregirlo, el modelo de embeddings no relaciona *Definici´on* con *Definición*. Se aplica un mapeo de reemplazos antes de indexar.
3. **Chunking sobre el documento completo.** Muchos ejemplos empiezan en una página y terminan en la siguiente. Chunkear por página separaba enunciado y resolución. Un mapa *offset → página* permite seguir citando la página.
4. **Chunks grandes (1500/300).** Los apuntes se organizan en "cajas" de definición y ejemplos con fórmulas. Con chunks de ~500 caracteres, la definición quedaba en un chunk y el ejemplo que la ilustra en otro.
5. **Curación del corpus.** *T12. Matrices y Sistemas* es la unión de *T12. Matrices 1C 2026* y *T13. Sistemas 1C 2026*. Indexar los tres llenaba el top-k de duplicados, así que se excluye (variable `EXCLUIR_PDFS`).
6. **Temperatura 0.** Hace la comparación CON vs. SIN RAG reproducible.

---

## 3. Estructura del repositorio

```
.
├── rag_tp.ipynb               # Notebook principal: pipeline completo + evaluación
├── rag_pdf_entrega_tp.py      # Versión de línea de comandos (consultas interactivas)
├── requirements.txt
├── docs/                      # PDFs del corpus
└── resultados/
    ├── evaluacion_respuestas.csv   # respuestas CON y SIN RAG, posición de la evidencia
    ├── evaluacion_corregida.csv    # misma tabla con la corrección manual
    └── experimento_prefijos.csv    # métricas de recuperación con y sin prefijos
```

---

## 4. Instalación

**Requisitos:** Python 3.10+, [Ollama](https://ollama.com) y unos 4 GB libres para los modelos.

```bash
# 1. Modelos
ollama pull gemma3
ollama pull nomic-embed-text

# 2. Entorno Python
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Corpus: copiar los PDFs a docs/
```

## 5. Ejecución

### Notebook (entregable principal)

```bash
jupyter notebook rag_tp.ipynb
```

Correr las celdas en orden. La primera ejecución indexa el corpus (unos minutos, según el hardware). Las siguientes reutilizan la base persistida en `chroma_tp/`. Para reindexar desde cero, usar `REINDEXAR = True`.

### Línea de comandos

```bash
python rag_pdf_entrega_tp.py "¿Qué es una recta tangente a una circunferencia?"
python rag_pdf_entrega_tp.py              # modo interactivo
python rag_pdf_entrega_tp.py --reindex    # reconstruye el índice
```

## 6. Ejemplos de uso

### Ejemplo 1: pregunta específica del apunte (P3)

**Pregunta:** ¿En qué orden conviene escribir los valores de verdad al armar la tabla de dos proposiciones?

**Fragmentos recuperados:** `T01. Lógica Proposicional.pdf` (pág. 7), entre otros.

**SIN RAG:**
> (pegar acá la respuesta sin RAG)

**CON RAG:**
> (pegar acá la respuesta con RAG)

**Comentario:** sin RAG, el modelo [resumí en una frase]. Con RAG reproduce la convención del apunte: V-V-F-F para la primera proposición y alternar V-F para la segunda.

### Ejemplo 2: pregunta fuera del corpus (P11)

**Pregunta:** ¿Cómo se calcula la derivada de un producto de funciones?

Las dos versiones respondieron con la regla del producto, (f·g)' = f'·g + f·g', y un ejemplo resuelto. La versión CON RAG **no advirtió que el tema no está en los apuntes** y no citó ninguna fuente, a pesar de que el prompt le pedía hacerlo. Ver la sección 8.

---

## 7. Evaluación

### Diseño

Se usaron 11 preguntas de tres tipos:

| Tipo | Cant. | Propósito |
|---|---|---|
| Específicas del apunte | 8 | Recursos y convenciones propios del curso: orden V-V-F-F en tablas de verdad, analogía de la bolsa para ∅ y {∅}, construcción de la tabla trigonométrica, "condiciones iniciales", etc. |
| Generales | 2 | Control: conceptos que el modelo ya conoce (número complejo, función inversa) |
| Fuera del corpus | 1 | Derivadas: el tema no está en los apuntes. Mide si el sistema se abstiene o alucina |

Se evalúan por separado las dos etapas del pipeline.

- **Recuperación (automática).** Para cada pregunta se definió una frase de *evidencia* que solo aparece en el fragmento que la responde. Se registra la posición del primer chunk recuperado que la contiene y se calculan **hit@k** (proporción de preguntas con la evidencia entre los primeros *k*) y **MRR** (promedio de 1/posición).
- **Generación (manual).** Cada respuesta se corrige contra la referencia del apunte: nota 0 a 10, presencia de alucinaciones y, en las respuestas CON RAG, si cita la fuente correcta.

### Resultados

Notas en escala de 0 a 10, corregidas a mano contra el apunte.

| Métrica | SIN RAG | CON RAG |
|---|---|---|
| Nota media, preguntas específicas (8) | 7,75 | 9,25 |
| Nota media, preguntas generales (2) | 7,00 | 10,00 |
| Nota media sin la pregunta fuera de corpus (10) | 7,60 | 9,40 |
| Se abstiene en la pregunta fuera de corpus | No | No |

El detalle de alucinaciones y citas por pregunta está en `resultados/evaluacion_corregida.csv` y en el resumen de la sección 7.3 del notebook.

| Recuperación | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| Sin prefijos (base) | 0,1 | 0,5 | 0,7 | 0,9 | 0,35 |
| Con prefijos `search_document:` / `search_query:` | 0,2 | 0,6 | 0,6 | 0,9 | 0,43 |

**Lectura de los resultados.** El retriever encuentra el fragmento correcto en 9 de 10 preguntas, pero rara vez lo ubica primero; por eso se usa k = 10. El RAG mejora la nota media de 7,6 a 9,4 porque el modelo responde con las definiciones y convenciones del apunte en lugar de definiciones genéricas. Los prefijos de `nomic-embed-text` mejoran el orden de los resultados (MRR de 0,35 a 0,43), aunque con 10 preguntas la diferencia no es concluyente.

---

## 8. Análisis crítico

### Limitaciones

- **Chunking por caracteres.** El corte cae en cualquier lugar: a mitad de palabra, de fórmula o de un ejemplo. El solapamiento lo mitiga, pero no lo resuelve.
- **Fórmulas LaTeX como texto plano.** Fracciones, raíces, matrices y sistemas se extraen desordenados (`x 2 + y2 = 1`, `Ö 1 2 è`). Eso degrada tanto el embedding como la respuesta en los temas más algebraicos (matrices, sistemas, ecuaciones).
- **Página citada aproximada.** Cada chunk se etiqueta con la página donde *empieza*. Si la respuesta está más adelante en el chunk, la cita apunta una o dos páginas antes. Por ejemplo, la propiedad de la recta tangente está en la pág. 45 de *Plano cartesiano*, pero el chunk que la contiene empieza en la 43.
- **Ruido de índices y encabezados.** El índice de cada apunte y el encabezado repetido en cada página ("Números Reales", "Plano Cartesiano") se indexan como contenido. Las consultas que nombran un tema pueden recuperar el índice en lugar de la explicación.
- **Embedding monolingüe en la práctica.** `nomic-embed-text` está entrenado mayormente en inglés. Funciona en castellano, pero un modelo multilingüe podría recuperar mejor.
- **Sin re-ranking.** Se usa directamente el orden por similitud vectorial.
- **Evaluación acotada.** Son 11 preguntas, corregidas a mano por una sola persona. Los resultados son indicativos, no estadísticamente concluyentes.

### Mejoras posibles

1. **Chunking estructural:** cortar en los límites de las "cajas" del apunte (Definición, Ejemplo, Observación, Propiedad) y de las secciones numeradas, en lugar de por cantidad de caracteres.
2. **Extracción consciente de fórmulas:** usar el código fuente LaTeX de los apuntes (si está disponible) o una herramienta como Nougat/Marker que convierta fórmulas a LaTeX.
3. **Limpieza:** eliminar índices, encabezados y pies de página antes de indexar.
4. **Búsqueda híbrida:** combinar embeddings con BM25 (palabras clave) para términos técnicos exactos como "Ruffini" o "De Moivre".
5. **Re-ranking** con un *cross-encoder* sobre los top-20 antes de quedarse con los 5 mejores.
6. **Citas exactas:** guardar el rango de páginas de cada chunk, no solo la inicial.
7. **Evaluación automática:** un conjunto más grande de preguntas y métricas de fidelidad al contexto (por ejemplo, con RAGAS).
8. **El modelo no siempre respeta el contexto.** En la pregunta fuera de corpus (derivadas), gemma3 respondió desde su conocimiento propio a pesar de la instrucción de abstenerse, sin citar fuentes ni advertir que el tema no estaba en los apuntes. Algo similar ocurrió en P8, donde respondió bien aunque el retriever no trajo el fragmento correcto. Mitigaciones posibles: un prompt de sistema más estricto, un umbral de distancia que evite llamar al LLM si no hay fragmentos cercanos, o un modelo más grande.
9. **Retrieval débil en temas con mucha notación.** La pregunta sobre clasificación de sistemas lineales (P8) no recuperó el fragmento correcto ni con prefijos: el texto de matrices y sistemas se extrae desordenado y eso afecta su embedding.

---

## 9. Sobre los datos

Los PDFs de la carpeta `docs/` son los apuntes teóricos del Trayecto Matemática de la Secretaría de Admisión del ITBA y se incluyen con fines académicos para que el trabajo sea reproducible. El archivo *T12. Matrices y Sistemas* está en la carpeta pero no se indexa, porque duplica el contenido de *T12. Matrices 1C 2026* y *T13. Sistemas 1C 2026* (ver la variable `EXCLUIR_PDFS` del notebook).
