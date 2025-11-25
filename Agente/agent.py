import os
# --- IMPORTACIONES ---
from langchain_community.llms import Ollama 
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate         
from langchain_core.output_parsers import StrOutputParser     
from langchain_core.runnables import RunnablePassthrough    

print("Iniciando Agente Profe Kepler")

#CONFIGURACIÓN
NOMBRE_PDF = "biologia.pdf"       
CARPETA_MEMORIA = "cerebro_kepler"  # Aquí se guardará la "memoria" del agente

#Config. IA
llm = Ollama(model="llama3", temperature=0.7)

# Cerebro para buscar información 
print("Cargando sistema de embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text") 

#Sistema de memoria inteligente 
def iniciar_vectorstore():
    """
    Esta función verifica si ya tenemos la memoria guardada en disco.
    Si existe, la carga (rápido). Si no, lee el PDF y la crea (lento la primera vez).
    """
    #Verifica si ya existe CARPETA_MEMORIA
    if os.path.exists(CARPETA_MEMORIA):
        print(f"✅ ¡Memoria encontrada en '{CARPETA_MEMORIA}'! Cargando instantáneamente...")
        # allow_dangerous_deserialization=True es necesario para cargar archivos locales propios
        return FAISS.load_local(CARPETA_MEMORIA, embeddings, allow_dangerous_deserialization=True)
    
    else:
        #Si no existe, la crea desde cero
        print("⚡ No se encontró memoria previa. Creando nueva base de datos...")
        print(f"   -> Leyendo {NOMBRE_PDF}...")
        
        #Validación de archivo
        if not os.path.exists(NOMBRE_PDF):
            print(f"\n[ERROR CRÍTICO] No encuentro el archivo '{NOMBRE_PDF}'.")
            print("Asegúrate de que el archivo PDF esté en la misma carpeta que este script.")
            exit()

        #Carga del PDF
        try:
            loader = PyMuPDFLoader(NOMBRE_PDF)
            documentos = loader.load()
        except Exception as e:
            print(f"[ERROR] Falló la carga del PDF: {e}")
            exit()

        if not documentos:
            print("[ERROR] El PDF parece vacío o no se pudo leer texto.")
            exit()

        print(f"   -> PDF cargado. Total de páginas: {len(documentos)}")

        #División del texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documentos)
        
        if len(chunks) == 0:
            print("[ERROR] No se pudo extraer texto. El PDF podría ser escaneado (imágenes).")
            exit()
            
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

#Definicion del prompt
system_prompt_kepler = """
Eres "Profe Kepler", un docente de Biología y Ciencias, paciente y riguroso. 

REGLAS ABSOLUTAS:
1.  **Metodología Socrática:** Si el alumno pregunta algo básico, responde con una pregunta guía.
2.  **REGLA DE ORO (RAG):** Basa tus respuestas EXCLUSIVAMENTE en el [CONTEXTO].
3.  **Citación:** Menciona que la información viene de tus libros.
4.  **Si no sabes:** Di "Esa información no está en mi libro de biología actual".
5.  **Tono:** Amable y académico.

[CONTEXTO]
{contexto}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_kepler),
    ("human", "{pregunta}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#Cadena rag
rag_chain = (
    {"contexto": retriever | format_docs, "pregunta": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

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
            print("¡Adiós! Sigue estudiando.")
            break
        
        if not pregunta_usuario.strip():
            continue

        print("Profe Kepler: (Consultando apuntes...)")
        respuesta = rag_chain.invoke(pregunta_usuario)
        print(f"\nProfe Kepler: {respuesta}")

    except KeyboardInterrupt:
        print("\nSesión finalizada.")
        break
    except Exception as e:
        print(f"\n[ERROR] Algo salió mal: {e}")