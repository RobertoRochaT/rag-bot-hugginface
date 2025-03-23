#   app.py
#   
#  Este archivo contiene el código de la API de Chatbot Laguna.
#
#  La API de Chatbot Laguna es una API RESTful que permite a los usuarios realizar consultas y obtener respuestas
#  Basadas en un modelo de lenguaje de Hugging Face y un vectorstore de Langchain.
#  Los usuarios pueden cargar documentos PDF, realizar consultas y 
#  obtener respuestas basadas en el contenido de los documentos.
#  También pueden obtener información sobre las conversaciones y documentos cargados.
#  Además, la API permite a los usuarios realizar consultas y obtener respuestas basadas 
#  en el contenido de páginas web dadas por el usuario.
#  
#  Contribuidores:
#
#  - Carlos Roberto Rocha Trejo el 22/03/2025 (
#    GitHub: https://github.com/RobertoRochaT
#    Linkedin: https://www.linkedin.com/in/carlosr-rocha
#  )
#  
#

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, Body
import os
import base64
import uuid
import certifi
import shutil
from pydantic import BaseModel
from base64 import b64decode
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime
from werkzeug.utils import secure_filename
from pymongo import MongoClient

from langchain_core.runnables import RunnableMap
from bson.objectid import ObjectId
from typing import List
import httpx
import requests

from bs4 import BeautifulSoup
from PIL import Image
from urllib.parse import urljoin, urlparse
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from transformers import pipeline


# Cargamos las variables de entorno
load_dotenv()

# Inicializamos la aplicación FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware, # Middleware para manejar las solicitudes de origen cruzado
    allow_origins=["*"],  # Permite todos los orígenes (cambiar en producción)
    allow_credentials=True, # Permite credenciales de origen cruzado
    allow_methods=["*"], # Permite todos los métodos de solicitud
    allow_headers=["*"], # Permite todas las cabeceras de solicitud
)

# Configuración de las claves de API

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Este es el modelo de lenguaje de Hugging Face y el que usaremos para el bot
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") # Esta es la clave de la API de Langchain
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") # Esta es la clave de trazado de Langchain
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN") # Esta es la clave de la API de Hugging Face

os.environ['SSL_CERT_FILE'] = certifi.where() # Establecer la ubicación del archivo de certificado SSL
# certifi es un paquete de python que proporciona un conjunto de certificados CA de confianza
# este nos sirve para evitar errores de certificado SSL al realizar solicitudes HTTPS

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Deshabilitar el paralelismo de tokenizadores para evitar errores de concurrencia

# Configuración de la carpeta temporal
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp') # Aqui creamos una carpeta temporal para almacenar archivos
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Configuración de MongoDB
MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
db = client.chatbotlaguna
conversations_collection = db.conversations
documents_collection = db.documents

# Configuración del vectorstore
vectorstore = Chroma( # Aqui creamos una base de datos de vectores para almacenar los documentos
    collection_name="multi_modal_rag", # Este es el modelo para generar 'embeddings' de los documentos
    # Un 'embedding' es una representación numérica de un documento que captura su significado semántico
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"), # Aqui usamos un modelo de lenguaje de Hugging Face para generar 'embeddings'
    persist_directory='./_temp' # Aqui indicamos la ubicacion donde se guardara la base de datos de vectores
)

store = InMemoryStore() # Creamos un almacenamiento en memoria para guardar datos temporalmente
# Este almacenamiento se utiliza para almacenar los documentos cargados en la base de datos de vectores
id_key = "doc_id" # Aqui indicamos la clave que se utilizara para identificar los documentos

# Aqui estamos configurando un recuperador de informacion 
retriever = MultiVectorRetriever( # Este es un recuperador de informacion en Langchain que se utiliza para recuperar documentos relevantes
    vectorstore=vectorstore, # Aqui 'vectorstore' es una base de datos vectorial (en este caso chroma) donde los documentos se almacenan como embedding
    docstore=store, # 'store' es un almacenamiento en memoria (InMemoryStore) donde se guardan los documentos originales junto con sus identificadores
    # Esto es util porque 'vectorstore' almacena solo embeddings, mientras que 'docstore' almacena los documentos originales
    id_key=id_key, # Aqui definimos la clave concreta 
    # Esto nos permite buscar en el vector store y luego recuperar el documento original desde el docstore
)

# Configuración del modelo de Hugging Face
llm = HuggingFaceEndpoint( # 'HuggingFaceEndPoint' es un punto final de Hugging Face que se utiliza para interactuar con los modelos de lenguaje de Hugging Face
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", # Especificamos el modelo este es un LLM optimizado para instrucciones
    temperature=0.5, # Con esto controlamos la aleatoridad del modelo 
    # Valores bajos (ej. 0.2) -> generan respuestas mas deterministicas
    # Valores altos (ej. 0.8) -> generan respuestas mas creativas
    # 0.5 es un valor intermedio que equilibra la aleatoriedad y la coherencia
    model_kwargs={"max_length": 512}  # Definimos la longitud maxima de la secuencia de tokens
)

def validate_documents(docs): # Esta es una funcion que usaremos mas adelante para validar los documentos
    for doc in docs:
        if not isinstance(doc, Document):
            raise ValueError("Invalid document format. Expected a `Document` object.")
        if not hasattr(doc, "page_content"):
            raise ValueError("Document is missing `page_content`.")
        if not hasattr(doc, "metadata"):
            raise ValueError("Document is missing `metadata`.")


# Función para cargar documentos existentes en el vectorstore
def load_existing_documents(): # Esta funcion es muy relevante para cargar los documentos ya existentes en el vectorstore
    try:
        documents = documents_collection.find({}) # Primero consultamos a MongoDB para recuperar la collection de documentos
        for doc in documents: # Luego iteramos sobre los documentos
            doc_ids = doc.get("doc_ids", []) # Extraemos los IDs de los documentos
            texts = doc.get("texts", []) # Lista de textos
            if doc_ids and texts: # Si existen IDs y textos
                summary_texts = [ # Generamos una lista de 'Documents' de langchain a partir de los textos
                    Document(page_content=text, metadata={id_key: doc_ids[i]}) for i, text in enumerate(texts) # Page_content es el texto del documento y metadata es un diccionario que contiene el ID del documento
                    # Esto es necesario para que podamos recuperar el documento original a partir del docstore
                ]
                validate_documents(summary_texts)  # Aqui validamos los documentos con la funcion anterior
                retriever.vectorstore.add_documents(summary_texts) # Agregamos los documentos al vectorstore
                retriever.docstore.mset(list(zip(doc_ids, summary_texts))) # Agregamos los documentos al docstore
        print("Existing documents loaded successfully.")
    except Exception as e:
        print(f"Error loading existing documents: {str(e)}")

# Cargar documentos existentes al iniciar el servidor
load_existing_documents()

# Función para parsear documentos esta nos ayudara mas adelante para extraer texto e imagenes de los documentos
# Debido a que nuestro bot permite a los usuarios cargar documentos PDF, necesitamos una forma de extraer texto e imágenes de estos documentos.

def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs: # Iteramos sobre los documentos y los clasificamos
        if isinstance(doc, Document): # Si el documento es un objeto 'Document' de LangChain, extraemos su contenido de la pagina
            text.append(doc.page_content)
            if "image" in doc.metadata:  
                b64.append(doc.metadata["image"]) # Si el documento tiene una imagen, la extraemos y la agregamos a la lista de imágenes
        elif isinstance(doc, str): # Si el documento es una cadena, lo agregamos a la lista de textos directamente
            text.append(doc)
        else: # Si el documento no es un objeto 'Document' ni una cadena, lanzamos un error
            raise ValueError(f"Unexpected document type: {type(doc)}")
    return {"images": b64, "texts": text}

# Función para construir el prompt
def build_prompt(kwargs):
    docs_by_type = kwargs["context"] # Lista de documentos por tipo (texto, tabla, imagen)
    user_question = kwargs["question"] # Pregunta del usuario

    context_text = "".join(docs_by_type["texts"]) #context_text = "".join(text_element.text for text_element in docs_by_type["texts"])
    # Aqui extraemos el texto de los documentos y lo concatenamos en una sola cadena
    prompt_template = f"""
    You are an AI assistant for Tec Laguna specialized in analyzing text and images.
    Use the following context to answer the question as accurately as possible.

    Context: {context_text}
    Question: {user_question}

    Provide a structured response with clear explanations.
    """
    # Aqui creamos una plantilla de prompt que incluye el contexto y la pregunta del usuario
    prompt_content = [{"type": "text", "text": prompt_template}] # Luego creamos una lista de contenido de prompt con el texto de la plantilla

    if len(docs_by_type["images"]) > 0: # Si hay imágenes en los documentos, las agregamos al prompt
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages( # Finalmente, creamos un ChatPromptTemplate a partir del contenido del prompt
        [
            HumanMessage(content=prompt_content),
        ]
    )


# Cadena de procesamiento
chain = RunnableMap({ # Aqui creamos una cadena de procesamiento que consta de varios pasos
    "context": retriever | RunnableLambda(parse_docs), # El primer paso es recuperar el contexto de los documentos y analizarlos
    "question": RunnablePassthrough(), # El segundo paso es pasar la pregunta del usuario
}) | RunnableLambda(build_prompt) | llm | StrOutputParser() # El tercer paso es construir el prompt, ejecutar el modelo de lenguaje y analizar la salida


chain_with_sources = { # Aqui creamos una cadena de procesamiento que incluye los documentos originales
                         "context": retriever | RunnableLambda(parse_docs), # El primer paso es recuperar el contexto de los documentos y analizarlos
                         "question": RunnablePassthrough(), # El segundo paso es pasar la pregunta del usuario
                     } | RunnablePassthrough().assign( # El tercer paso es asignar los documentos originales al contexto
    response=(
            RunnableLambda(build_prompt) # El cuarto paso es construir el prompt
            | llm # El quinto paso es ejecutar el modelo de lenguaje
            | StrOutputParser() # El sexto paso es analizar la salida
    )
)

# chain y chain_with_sources son lo mismo, pero chain_with_sources incluye los documentos originales en el contexto
# Esto es útil para proporcionar información adicional al modelo de lenguaje y mejorar la calidad de las respuestas


# Modelo de solicitud
# se utiliza para modelar una solicitud de consulta, 
# específicamente con el fin de procesar las preguntas de los usuarios 
# dentro de un sistema de conversación. 
class QueryRequest(BaseModel): # 'BaseModel' es una clase de Pydantic que se utiliza para definir modelos de datos
    question: str # Aqui definimos la pregunta del usuario
    conversation_name: str = "Default Conversation" # Aqui definimos el nombre de la conversacion

translate_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es") # Aqui creamos un pipeline de traduccion para traducir las respuestas del modelo de lenguaje

# Empezamos con las rutas de la API
@app.get("/") # Esta es la ruta raiz de la API, nos devuelve un mensaje de bienvenida
# Con esta ruta, podemos verificar si la API está en funcionamiento y obtener un mensaje de bienvenida
async def read_root():
    return {"message": "Bienvenido al Chatbot Laguna v1.0"} # Aqui devolvemos un mensaje de bienvenida

# Ruta para obtener los documentos
@app.get('/get_documents') # Esta es la ruta para obtener los documentos cargados en la base de datos
# Esta ruta nos permite obtener una lista de todos los documentos cargados en la base de datos
# Nos servira mucho en el frontend para mostrar los documentos cargados
async def read_root():
    try:
        # Obtener todos los documentos de la colección, incluyendo el _id y campos relevantes
        documents = list(documents_collection.find({}, {"_id": 1, "file_path": 1, "doc_ids": 1, "texts": 1, "tables": 1, "images": 1, "upload_date": 1}))
        
        # Convertir ObjectId a string para que sea serializable
        for doc in documents:
            doc["_id"] = str(doc["_id"])
        
        # Devolver los documentos
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para eliminar un documento
@app.post('/delete_document')
async def delete_document(id: str): # Esta es la ruta para eliminar un documento de la base de datos
    try:
        # Buscar el documento en la colección de MongoDB por su _id
        document = documents_collection.find_one({"_id": ObjectId(id)})
        
        if not document:
            raise HTTPException(status_code=404, detail="Documento no encontrado")

        # Obtener lista de IDs relacionados y ruta del archivo
        doc_ids = document.get("doc_ids", [])
        file_path = document.get("file_path")

        # Eliminar del vectorstore y almacenamiento en memoria si existen IDs
        if doc_ids:
            retriever.vectorstore.delete(doc_ids)
            retriever.docstore.mdelete(doc_ids)

        # Eliminar el archivo del sistema de archivos si existe
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        # Eliminar la entrada de la base de datos
        documents_collection.delete_one({"_id": ObjectId(id)})

        return {"message": f"Documento con ID '{id}' eliminado correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
  
# Ruta para incrustar archivos PDF
@app.post("/embed") # Esta es la ruta para incrustar archivos PDF en la base de datos
async def embed_pdf(file: UploadFile = File(...)): # Aqui usamos el modelo de datos 'UploadFile' de FastAPI para cargar el archivo PDF
    if not file.filename.endswith(".pdf"): # Verificamos si el archivo es un PDF
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

    # Guardar el archivo en la carpeta de documentos
    file_path = f"./documents/{secure_filename(file.filename)}"
    with open(file_path, "wb") as f: # Guardamos el archivo en el sistema de archivos
        shutil.copyfileobj(file.file, f) # Copiamos el archivo en el sistema de archivos

    # Procesar el PDF
    chunks = partition_pdf( # Aqui llamamos a la funcion 'partition_pdf' para procesar el PDF
        filename=file_path, # Aqui pasamos la ruta del archivo que se va a procesar
        infer_table_structure=True, # Esta opcion permite inferir la estructura de las tablas en el PDF
        # Esto es util para extraer tablas de manera mas efectiva 
        strategy="hi_res", # Esto indica que usar una estrategia de alta resolucion para extraer el contenido del PDF
        # Esto es util para extraer texto e imagenes de alta calidad
        extract_image_block_types=["Image"], # Aqui indicamos que solo extraeremos bloques de imagen
        # Esto es util para extraer solo las imagenes del PDF
        extract_image_block_to_payload=True, # Esto significa que las imágenes extraídas se incluirán en el "payload" de la respuesta. 
        #Es decir, se devolverán junto con los fragmentos de texto, lo que es util para el análisis multimodal (texto e imágenes).
        chunking_strategy="by_title", # Aqui definimos como se dividira el PDF en fragmentos
        max_characters=10000, # Aqui definimos el numero maximo de caracteres por fragmento
        combine_text_under_n_chars=2000, # Aqui definimos el numero de caracteres para combinar texto
        new_after_n_chars=6000, # Aqui definimos el numero de caracteres para crear un nuevo fragmento
    )

    # Separar textos, tablas e imágenes
    texts = [chunk for chunk in chunks if "CompositeElement" in str(type(chunk))]
    # Aqui extraemos los textos de los fragmentos
    tables = [chunk for chunk in chunks if "Table" in str(type(chunk))] # Aqui extraemos las tablas de los fragmentos
    # Aqui extraemos las imagenes de los fragmentos
    images = get_images_base64(chunks)

    # Agregar textos al vectorstore
    doc_ids = [str(uuid.uuid4()) for _ in texts] # Aqui creamos un 'doc_id'
    summary_texts = [
        Document(page_content=chunk.text, metadata={id_key: doc_ids[i]}) for i, chunk in enumerate(texts) #
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Guardar la información del documento en MongoDB
    document_data = {
        "file_path": file_path,
        "doc_ids": doc_ids,
        "texts": [chunk.text for chunk in texts],
        "tables": [chunk.text for chunk in tables],
        "images": images,
        "upload_date": datetime.now()
    }
    documents_collection.insert_one(document_data)

    return {"message": "PDF procesado y datos incrustados correctamente"}

# Ruta para incrustar multiples pdfs
@app.post("/embed_multiple") # Esto es similar a la ruta anterior, pero permite a los usuarios cargar varios archivos PDF a la vez
async def embed_pdfs(files: List[UploadFile] = File(...)):
    results = []
    
    for file in files:
        if not file.filename.endswith(".pdf"):
            continue  # Ignorar archivos que no sean PDF

        # Guardar el archivo en la carpeta de documentos
        file_path = f"./documents/{secure_filename(file.filename)}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Procesar el PDF
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )

        # Separar textos, tablas e imágenes
        texts = [chunk for chunk in chunks if "CompositeElement" in str(type(chunk))]
        tables = [chunk for chunk in chunks if "Table" in str(type(chunk))]
        images = get_images_base64(chunks)

        # Agregar textos al vectorstore
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=chunk.text, metadata={"doc_id": doc_ids[i]}) for i, chunk in enumerate(texts)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

        # Guardar la información del documento en MongoDB
        document_data = {
            "file_path": file_path,
            "doc_ids": doc_ids,
            "texts": [chunk.text for chunk in texts],
            "tables": [chunk.text for chunk in tables],
            "images": images,
            "upload_date": datetime.now()
        }
        documents_collection.insert_one(document_data)

        results.append({"file": file.filename, "message": "PDF procesado correctamente"})

    if not results:
        raise HTTPException(status_code=400, detail="Ningún archivo válido fue procesado.")

    return {"processed_files": results}

async def classify_intent(question: str) -> str:
    try:
        url = "https://kuzeee-intentclassifier.hf.space/classify"
        data = {"sentence": question}  # Cambia "question" a "sentence"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data)
            response.raise_for_status()
        result = response.json()
        print(f"Response from intent classifier: {result}")  # Log the response
        # Extraer la intención de la respuesta
        if "classification" in result and len(result["classification"]) > 0:
            return result["classification"][0][0]  # Devuelve la primera intención
        return "unknown"
    except Exception as e:
        print(f"Error classifying intent: {str(e)}")
        return "unknown"

@app.post("/query")
async def query(request: QueryRequest):
    status = "Failed"
    conversation = request.conversation_name
    try:
        # Clasificar la intención de la pregunta
        intent = await classify_intent(request.question)
        print(intent)
        # Si la intención no es relacionada con la universidad, devolver un mensaje de error
        if intent != "university_related":  # Asegúrate de que "university_related" sea la etiqueta correcta
            conversation_data = {
                "conversation_name": request.conversation_name,
                "question": request.question,
                "response": "Lo siento, solo puedo responder preguntas relacionadas con la universidad.   intent:" + intent,
                "timestamp": datetime.now(),
                "status": "Failed"
            }
            conversations_collection.insert_one(conversation_data)
            return {"response": "Lo siento, solo puedo responder preguntas relacionadas con la universidad.   intent:" + intent}
        
        # Si la intención es relacionada con la universidad, continuar con el procesamiento normal
        num_docs = len(retriever.vectorstore.get()["ids"])
        if num_docs == 0:
            raise HTTPException(status_code=400, detail="No documents available for retrieval. Please upload documents first.")

        print(f"Number of documents in vectorstore: {num_docs}")  # Log the number of documents

        n_results = min(4, num_docs)  # Ensure n_results does not exceed available documents
        retriever.vectorstore._collection.max_results = n_results

        retrieved_docs = retriever.get_relevant_documents(request.question)
        print(f"Retrieved documents: {retrieved_docs}")  # Log the retrieved documents

        response = chain.invoke(request.question)

        # Guardar la conversación en MongoDB
        conversation_data = {
            "conversation_name": request.conversation_name,
            "question": request.question,
            "response": response,
            "timestamp": datetime.now(),
            "status": status
        }
        conversations_collection.insert_one(conversation_data)

        if response is not None:
            conversation = "Conversación Exitosa"
            status = "Success"

        if "The context provided does not include any specific information" in response or "I cannot directly answer your question based on the provided context" in response:
            conversation = "Conversación Fallida"
            status = "Failed"

        conversations_collection.update_one({"question": request.question}, {"$set": {"conversation_name": conversation}})
        conversations_collection.update_one({"question": request.question}, {"$set": {"status": status}})

        return {"response": response + "   intent:" + intent}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during query processing: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para ver las consultas que se han realizado
@app.get("/queries")
async def get_queries(conversation_name: str = None, page: int = 1, page_size: int = 10):
    try:
        # Crear el filtro de búsqueda
        filter_query = {"conversation_name": conversation_name} if conversation_name else {}

        # Paginación
        skip = (page - 1) * page_size
        limit = page_size

        # Obtener las consultas desde MongoDB
        consultas = list(conversations_collection.find(filter_query, {"_id": 0}).skip(skip).limit(limit))

        if not consultas:
            raise HTTPException(status_code=404, detail="No queries found for the given parameters.")

        # Devuelvo las consultas con información adicional
        response = {
            "page": page,
            "page_size": page_size,
            "total": conversations_collection.count_documents(filter_query),
            "queries": consultas,
        }

        return response
    except Exception as e:
        # Error general con una descripción detallada
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Ruta para obtener una conversación específica
@app.get("/conversations")
async def get_conversations():
    try:
        # Obtener todas las conversaciones de la colección
        conversations = list(conversations_collection.find({}, {"_id": 0}))
        
        # Devolver las conversaciones
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSockets  
connections = []
# Aqui creamos una lista de conexiones para almacenar los clientes conectados al WebSocket
@app.websocket("/ws") # Esta es la ruta para el WebSocket
async def websocket_endpoint(websocket: WebSocket): # Aqui usamos el modelo de datos 'WebSocket' de FastAPI para manejar las conexiones WebSocket
    await websocket.accept()
    connections.append(websocket) # Agregamos la conexión a la lista de conexiones
    try:
        while True: # Aqui creamos un bucle infinito para recibir y enviar mensajes
            data = await websocket.receive_text() # Recibimos un mensaje del cliente
            for conn in connections: # Iteramos sobre las conexiones y enviamos el mensaje a cada cliente
                await conn.send_text(data) # Enviamos el mensaje al cliente
    except Exception as e:
        print(f"Error en WebSocket: {e}")
    finally:
        connections.remove(websocket)


# Función para obtener imágenes en base64
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

# Este es un ejemplo de cómo se puede usar la API de Chatbot Laguna para obtener respuestas basadas en el contenido de una página web.
# Aqui usaremos el /get_links para obtener los enlaces que esten vinculados a una pagina web
@app.post('/get_links')
async def get_links(request: dict = Body(...)):
    url = request.get('url')

    if not url:
        raise HTTPException(status_code=400, detail="URL no proporcionada")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

    soup = BeautifulSoup(response.text, 'html.parser')
    links = []

    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            links.append(href)
    return {"links": links}

# Funcion apra scrapear la pagina web
def scrape_webpage(url):
    print(f"Scraping URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping URL: {str(e)}")

# Function to extract text
def extract_text(soup):
    print("Extracting text...")
    try:
        text = soup.get_text(separator="\n")
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")


# Función para extraer las imagenes de la página web
def extract_images(soup, base_url):
    print("Extracting images...")
    try:
        images = []
        for img in soup.find_all('img'):
            img_url = img.get('src')
            if img_url and img.find_parent('div', class_='row'):
                full_url = urljoin(base_url, img_url)
                images.append(full_url)
        print(f"Found {len(images)} images in 'div' with class 'row'.")
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting images: {str(e)}")
    
# Función para extraer tablas
def extract_tables(soup):
    print("Extracting tables...")
    try:
        tables = []
        for table in soup.find_all('table'):
            rows = []
            for row in table.find_all('tr'):
                cells = []
                for cell in row.find_all(['th', 'td']):
                    cells.append(cell.get_text(strip=True))
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
        return tables
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting tables: {str(e)}")

# Función para descargar imágenes
def download_images(image_urls, folder="images"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    img_path = folder
    downloaded_images = []
    for img_url in image_urls:
        try:
            response = requests.get(img_url, stream=True)
            if response.status_code == 200:
                img_name = os.path.basename(img_url)
                img_path = os.path.join(folder, img_name)
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded image: {img_name}")
                downloaded_images.append(img_path)
            else:
                print(f"Failed to download: {img_url}")
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")
    return downloaded_images

# Función para generar PDF
def generate_pdf(text, images, tables, output_file="output.pdf"):
    print("Generating PDF...")
    try:
        pdf = SimpleDocTemplate(output_file, pagesize=A4)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            name='TitleStyle',
            parent=styles['Title'],
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
        body_style = ParagraphStyle(
            name='BodyStyle',
            parent=styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )

        elements = []

        # Add title
        elements.append(Paragraph("Webpage Scraped Content", title_style))
        elements.append(Spacer(1, 12))

        # Add text
        paragraphs = text.split('\n')
        for para in paragraphs:
            if para.strip():
                elements.append(Paragraph(para.strip(), body_style))
                elements.append(Spacer(1, 6))

        # Add tables
        for table_data in tables:
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

        # Add images
        for img_path in images:
            try:
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img = ReportLabImage(img_path, width=5*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 12))
                else:
                    print(f"Skipping unsupported image format: {img_path}")
            except Exception as e:
                print(f"Error adding image {img_path} to PDF: {e}")

        # Build PDF
        pdf.build(elements)
        print(f"PDF generated successfully: {output_file}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

# Función para enviar el PDF al endpoint /embed
async def send_pdf_to_embed(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: El archivo {pdf_path} no existe.")
        raise HTTPException(status_code=500, detail="El archivo PDF no se encuentra.")

    try:
        with open(pdf_path, "rb") as pdf_file:
            files = {"file": (pdf_path, pdf_file, "application/pdf")}

            print(f"Enviando archivo {pdf_path} a /embed...")

            async with httpx.AsyncClient(timeout=60) as client:  # Usa un cliente async
                response = await client.post("http://localhost:8500/embed", files=files)

            response.raise_for_status()  # Verifica si el código de estado no es 2xx

            print(f"Respuesta de /embed: {response.json()}")  # Log de la respuesta

            return response.json()

    except httpx.TimeoutException:
        print("Error: La solicitud a /embed ha excedido el tiempo de espera.")
        raise HTTPException(status_code=500, detail="Tiempo de espera excedido al enviar PDF a /embed")

    except httpx.RequestError as e:
        print(f"Error al enviar PDF a /embed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al conectar con /embed: {str(e)}")
    
class LinkRequest(BaseModel):
    links: List[str]
# Ruta para scrapear y generar PDF
@app.post("/scrape_and_embed")
async def scrape_and_embed(request: LinkRequest):  # <- Usa el modelo
    links = request.links  # Accede a la lista
    print(f"Received links: {links}")
    results = []
        
    # Determinar la base_url a partir de los enlaces proporcionados
    base_url = None
    for link in links:
        if link.startswith("http"):
            parsed_url = urlparse(link)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            break  # Usamos la primera URL completa como base
    
    if not base_url:
        return {"error": "No valid base URL found in the provided links."}
    
    print(f"Base URL: {base_url}")

    print(f"Scraping links: {links}")
    
    for link in links:
        try:
            # Construir la URL completa si es relativa
            url = link if link.startswith("http") else f"{base_url}{link}"

            # Scrapear la página web
            soup = scrape_webpage(url)

            # Extraer contenido
            text = extract_text(soup)
            images = extract_images(soup, url)
            tables = extract_tables(soup)

            # Descargar imágenes
            downloaded_images = download_images(images)

            # Generar un nombre único para el PDF
            current_date = datetime.now().strftime("%Y-%m-%d")
            pdf_path = f"./_temp/scraped_content_{current_date}_{link.replace('/', '_')}.pdf"

            # Generar PDF
            generate_pdf(text, downloaded_images, tables, pdf_path)

            # Enviar el PDF al endpoint /embed
            embed_response = await send_pdf_to_embed(pdf_path)

            # Limpiar archivos temporales
            for img_path in downloaded_images:
                os.remove(img_path)
            os.remove(pdf_path)

            results.append({
                "link": link,
                "status": "success",
                "pdf_path": pdf_path,
                "embed_response": embed_response
            })
        except Exception as e:
            results.append({
                "link": link,
                "status": "failed",
                "error": str(e)
            })

    return {"results": results}


# Crear la carpeta de documentos si no existe
if not os.path.exists("./documents"):
    os.makedirs("./documents")

# Iniciar la aplicación
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
