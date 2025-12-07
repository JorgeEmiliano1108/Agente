import os
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import json
import csv
import sys
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

print("Iniciando Agente Profe Kepler")

#CONFIGURACIÓN
# Permitir especificar el PDF por argumento o variable de entorno
parser = argparse.ArgumentParser(description="Profe Kepler - Agente de consulta de un PDF de biología")
parser.add_argument("--pdf", "-p", help="Ruta al archivo PDF que contiene el libro (por defecto 'biologia.pdf')", default=None)
parser.add_argument("--build-only", action="store_true", help="Construir/actualizar la memoria a partir del PDF y salir")
parser.add_argument("--dry-run", action="store_true", help="No llamar al LLM; solo mostrar el contexto recuperado")
parser.add_argument("--retries", type=int, default=2, help="Número máximo de reintentos para llamadas al LLM")
parser.add_argument("--log-file", default="logs/agent.log", help="Ruta del archivo de log")
args = parser.parse_args()

# Configurar logging
log_path = Path(args.log_file)
log_path.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("profe_kepler")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler(log_path, encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

# Prioridad: argumento CLI > variable de entorno PDF_PATH > nombre por defecto
NOMBRE_PDF = args.pdf or os.getenv("PDF_PATH") or "biologia.pdf"
CARPETA_MEMORIA = "cerebro_kepler"  # Aquí se guardará la "memoria" del agente

#Config. IA
llm = OllamaLLM(model="llama3", temperature=0.7)

# Cerebro para buscar información 
print("Cargando sistema de embeddings...")
logger.info("Cargando sistema de embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text") 

#Sistema de memoria inteligente 
def iniciar_vectorstore():
    """
    Esta función verifica si ya tenemos la memoria guardada en disco.
    Si existe, la carga (rápido). Si no, lee el PDF y la crea (lento la primera vez).
    """
    #Verifica si ya existe CARPETA_MEMORIA
    if os.path.exists(CARPETA_MEMORIA):
        msg = f"✅ ¡Memoria encontrada en '{CARPETA_MEMORIA}'! Cargando instantáneamente..."
        print(msg)
        logger.info(msg)
        # allow_dangerous_deserialization=True es necesario para cargar archivos locales propios
        return FAISS.load_local(CARPETA_MEMORIA, embeddings, allow_dangerous_deserialization=True)
    
    else:
        #Si no existe, la crea desde cero
        msg = "⚡ No se encontró memoria previa. Creando nueva base de datos..."
        print(msg)
        logger.info(msg)
        print(f"   -> Leyendo {NOMBRE_PDF}...")
        logger.info(f"   -> Leyendo {NOMBRE_PDF}...")
        
        #Validación de archivo
        if not os.path.exists(NOMBRE_PDF):
            logger.error(f"No encuentro el archivo '{NOMBRE_PDF}'.")
            print(f"ERROR: No encuentro el archivo '{NOMBRE_PDF}'. Especifica --pdf o pon PDF en la carpeta.")
            sys.exit(1)

        #Carga del PDF
        try:
            loader = PyMuPDFLoader(str(NOMBRE_PDF))
            documentos = loader.load()
        except Exception as e:
            logger.error(f"Error cargando PDF: {e}")
            print(f"Error cargando PDF: {e}")
            raise

        if not documentos:
            logger.error("No se extrajo texto del PDF (documentos vacío). Revisa el PDF o el loader.")
            print("No se extrajo texto del PDF. Saliendo.")
            sys.exit(1)

        print(f"   -> PDF cargado. Total de páginas/documentos: {len(documentos)}")

        #División del texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documentos)
        
        if len(chunks) == 0:
            logger.error("La división de texto devolvió 0 fragments. Revisa el splitter.")
            print("Error: no se obtuvieron fragmentos del PDF.")
            sys.exit(1)
            
        print(f"   -> Conocimiento dividido en {len(chunks)} fragmentos.")

        #Creación del índice vectorial
        print("   -> Vectorizando datos (esto puede tardar unos minutos la primera vez)...")
        # Aquí usamos nomic-embed-text, que es rápido y no saturará tu RAM
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        #Save in disk
        vector_store.save_local(CARPETA_MEMORIA)
        print(f"💾 ¡Memoria guardada exitosamente en '{CARPETA_MEMORIA}'!")
        
        return vector_store

#Inicializamos la memoria
vector_store = iniciar_vectorstore()
retriever = vector_store.as_retriever(search_kwargs={"k": 4}) 
print("Recuperador listo.")
logger.info("Recuperador listo.")

# Si el usuario solo quiere construir la memoria, salir aquí
if args.build_only:
    logger.info("--build-only activado: memoria construida/actualizada. Saliendo.")
    print("Memoria construida/actualizada. Saliendo (--build-only).")
    exit(0)

#Definicion del prompt
system_prompt_kepler = """
Eres "Profe Kepler", un docente de Biología y Ciencias, paciente y riguroso. 

REGLAS ABSOLUTAS:
1.  Metodología Socrática: Si el alumno pregunta algo básico, responde con una pregunta guía.
2.  REGLA DE ORO (RAG): Basa tus respuestas EXCLUSIVAMENTE en el [CONTEXTO].
3.  Citación: Menciona que la información viene de tus libros.
4.  Si no sabes: Di "Esa información no está en mi libro de biología actual".
5.  Tono: Amable y académico.

[CONTEXTO]
{contexto}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_kepler),
    ("human", "{pregunta}")
])

def format_docs(docs):
    """Convierte una lista de Document en un texto plano concatenado."""
    try:
        parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", doc.metadata.get("source_id", f"doc_{i}"))
            content = getattr(doc, "page_content", str(doc))
            parts.append(f"--- Fuente: {source} ---\n{content}")
        return "\n\n".join(parts)
    except Exception:
        return "\n\n".join(str(doc) for doc in docs)

def extract_sources(docs):
    """Extrae una lista breve de fuentes/metadata desde los documentos recuperados."""
    sources = []
    try:
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source") or doc.metadata.get("source_id") or doc.metadata.get("filename") or f"doc_{i}"
            sources.append(src)
        # Devolver solo entradas únicas y primeras 6
        return list(dict.fromkeys(sources))[:6]
    except Exception:
        return []

def save_history(pregunta, sources, respuesta, dry_run=False):
    """Guarda la consulta en CSV y JSONL dentro de `logs/`.

    Campos: timestamp, pregunta, fuentes (lista), dry_run (bool), respuesta (texto corto)
    """
    try:
        ts = datetime.utcnow().isoformat() + "Z"
        logdir = Path(args.log_file).parent
        # JSONL
        jsonl_path = logdir / "history.jsonl"
        entry = {
            "timestamp": ts,
            "pregunta": pregunta,
            "fuentes": sources,
            "dry_run": bool(dry_run),
            "respuesta": respuesta,
        }
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # CSV (append, create header if needed)
        csv_path = logdir / "history.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", encoding="utf-8", newline="") as cf:
            writer = csv.writer(cf)
            if write_header:
                writer.writerow(["timestamp", "pregunta", "fuentes", "dry_run", "respuesta"])
            writer.writerow([ts, pregunta, ";".join(sources), str(bool(dry_run)), respuesta[:200]])
    except Exception as e:
        logger.warning(f"No se pudo guardar el historial: {e}")

def call_llm(prompt_text, max_retries=2, backoff_factor=1.5):
    """Intentos robustos para llamar al LLM y devolver texto.

    Prueba varios métodos comunes y extrae texto si la respuesta es un objeto complejo.
    """
    attempt = 0
    last_exception = None
    while attempt <= max_retries:
        try:
            # 1) Intentar .predict (suele devolver str)
            if hasattr(llm, "predict"):
                out = llm.predict(prompt_text)
                if isinstance(out, str):
                    return out
        except Exception as e:
            last_exception = e
            logger.debug(f"predict failed: {e}")

        # 2) Intentar .generate (objeto complejo)
        try:
            if hasattr(llm, "generate"):
                res = llm.generate([prompt_text])
                # intentar extraer texto de la estructura
                if hasattr(res, "generations"):
                    gens = res.generations
                    if gens and len(gens) > 0 and len(gens[0]) > 0:
                        text = gens[0][0].text if hasattr(gens[0][0], "text") else str(gens[0][0])
                        return text
                # fallback: string representation
                return str(res)
        except Exception as e:
            last_exception = e
            logger.debug(f"generate failed: {e}")

        # 3) Intentar __call__ / __call__-like
        try:
            if hasattr(llm, "__call__"):
                res = llm(prompt_text)
                if isinstance(res, str):
                    return res
                # some LLMs return dict-like
                if isinstance(res, dict) and "text" in res:
                    return res["text"]
                return str(res)
        except Exception as e:
            last_exception = e
            logger.debug(f"__call__ failed: {e}")

        # si no es el último intento, esperar backoff
        attempt += 1
        if attempt <= max_retries:
            sleep = backoff_factor * (2 ** (attempt - 1))
            logger.info(f"Reintentando LLM en {sleep:.1f}s (intento {attempt}/{max_retries})")
            time.sleep(sleep)

    # Si llegamos aquí, todos los intentos fallaron
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_exception}")

def obtener_respuesta(pregunta):
    """Flujo RAG simple: recuperar, formatear, construir prompt y llamar al LLM."""
    # Recuperar documentos relevantes: probar métodos comunes en orden
    docs = None
    errors = []
    for name in ("get_relevant_documents", "retrieve", "get_documents", "similarity_search"):
        try:
            if hasattr(retriever, name):
                func = getattr(retriever, name)
                # muchos retrievers esperan la pregunta como primer arg
                docs = func(pregunta)
                if docs:
                    break
        except Exception as e:
            errors.append((name, str(e)))

    if docs is None:
        # Intentar usar el vector_store directamente como fallback
        try:
            docs = vector_store.similarity_search(pregunta, k=4)
        except Exception as e:
            logger.error(f"Fallback similarity_search falló: {e}")
            docs = []

    contexto = format_docs(docs)

    # Extraer fuentes y registrar la consulta
    sources = extract_sources(docs)
    logger.info(f"Consulta: {pregunta} | Fuentes: {sources}")

    # Si se pidió dry-run, devolver el contexto sin llamar al LLM
    if args.dry_run:
        snippet = contexto[:1500]
        logger.info("Modo dry-run activado: no se llama al LLM")
        result = f"[DRY RUN] Contexto recuperado (primeros 1500 chars):\n{snippet}\n\nFuentes: {sources}"
        try:
            save_history(pregunta, sources, result, dry_run=True)
        except Exception:
            pass
        return result

    # Construir prompt final: usar la plantilla del sistema + pregunta
    system_block = system_prompt_kepler.format(contexto=contexto)
    # Prefijar fuentes explícitamente para cumplir la regla de citación
    fuentes_line = ", ".join(sources) if sources else "(fuente no especificada)"
    final_prompt = f"Fuentes: {fuentes_line}\n\n{system_block}\n\nPregunta: {pregunta}"

    try:
        respuesta = call_llm(final_prompt, max_retries=args.retries)
        try:
            save_history(pregunta, sources, respuesta, dry_run=False)
        except Exception:
            pass
        return respuesta
    except Exception as e:
        logger.error(f"Fallo en llamada al LLM: {e}")
        # Devolver contexto como fallback informativo
        snippet = contexto[:1500]
        fallback = f"No pude generar respuesta desde el LLM: {e}\n\nContexto recuperado:\n{snippet}\n\nFuentes: {sources}"
        try:
            save_history(pregunta, sources, fallback, dry_run=False)
        except Exception:
            pass
        return fallback

print("\n" + "="*50)
print("🤖 PROFE KEPLER ESTÁ LISTO")
print(f"📚 Libro activo: {NOMBRE_PDF}")
print(" Consejo: Si cambias el PDF, borra la carpeta 'cerebro_kepler' para regenerarla.")
print("="*50 + "\n")

#Bucle de chat
print("¡Hola! Soy Profe Kepler. ¿Qué duda de Biología tienes hoy?")
print("(Escribe 'salir' para terminar)")

while True:
    try:
        pregunta_usuario = input("\nTú: ")
        if pregunta_usuario.lower() in ["salir", "exit"]:
            print("Adiós.")
            break

        if not pregunta_usuario.strip():
            continue

        print("Profe Kepler: (Consultando apuntes...)")
        respuesta = obtener_respuesta(pregunta_usuario)
        print(f"\nProfe Kepler: {respuesta}")

    except KeyboardInterrupt:
        print("\nSesión finalizada.")
        break
    except Exception as e:
        print(f"\n[ERROR] Algo salió mal: {e}")