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
from urllib.parse import urljoin
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from transformers import pipeline

load_dotenv()

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (cambiar en producción)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de las claves de API

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuración de la carpeta temporal
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Configuración de MongoDB
MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
db = client.chatbotlaguna
conversations_collection = db.conversations
documents_collection = db.documents

# Configuración del vectorstore
vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
    persist_directory='./_temp'
)

store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Configuración del modelo de Hugging Face
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,  
    model_kwargs={"max_length": 512} 
)

def validate_documents(docs):
    for doc in docs:
        if not isinstance(doc, Document):
            raise ValueError("Invalid document format. Expected a `Document` object.")
        if not hasattr(doc, "page_content"):
            raise ValueError("Document is missing `page_content`.")
        if not hasattr(doc, "metadata"):
            raise ValueError("Document is missing `metadata`.")


# Función para cargar documentos existentes en el vectorstore
def load_existing_documents():
    try:
        documents = documents_collection.find({})
        for doc in documents:
            doc_ids = doc.get("doc_ids", [])
            texts = doc.get("texts", [])
            if doc_ids and texts:
                summary_texts = [
                    Document(page_content=text, metadata={id_key: doc_ids[i]}) for i, text in enumerate(texts)
                ]
                validate_documents(summary_texts)  # Validate documents
                retriever.vectorstore.add_documents(summary_texts)
                # Store Document objects in the docstore
                retriever.docstore.mset(list(zip(doc_ids, summary_texts)))
        print("Existing documents loaded successfully.")
    except Exception as e:
        print(f"Error loading existing documents: {str(e)}")
# Cargar documentos existentes al iniciar el servidor
load_existing_documents()

# Función para parsear documentos
def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        if isinstance(doc, Document):
            # If the doc is a Document object, extract its page_content
            text.append(doc.page_content)
        elif isinstance(doc, str):
            # If the doc is a string, use it directly
            text.append(doc)
        else:
            raise ValueError(f"Unexpected document type: {type(doc)}")
    return {"images": b64, "texts": text}

# Función para construir el prompt
def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = "".join(docs_by_type["texts"]) #context_text = "".join(text_element.text for text_element in docs_by_type["texts"])

    prompt_template = f"""
    Answer the question solely based on the following context, which may include text, tables, and images.
    Provide your answer clearly and concisely.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


# Cadena de procesamiento
chain = RunnableMap({
    "context": retriever | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
}) | RunnableLambda(build_prompt) | llm | StrOutputParser()


chain_with_sources = {
                         "context": retriever | RunnableLambda(parse_docs),
                         "question": RunnablePassthrough(),
                     } | RunnablePassthrough().assign(
    response=(
            RunnableLambda(build_prompt)
            | llm
            | StrOutputParser()
    )
)


# Modelo de solicitud
class QueryRequest(BaseModel):
    question: str
    conversation_name: str = "Default Conversation"

translate_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

@app.get("/")
async def read_root():
    return {"message": "Bienvenido al Chatbot Laguna v1.0"}

# Ruta para obtener los documentos
@app.get('/get_documents')
async def read_root():
    try:
        # Obtener todos los documentos de la colección, incluyendo el _id
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
async def delete_document(id: str):
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
@app.post("/embed")
async def embed_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

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
        Document(page_content=chunk.text, metadata={id_key: doc_ids[i]}) for i, chunk in enumerate(texts)
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
@app.post("/embed_multiple")
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

# Ruta para realizar consultas

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

@app.get("/conversations")
async def get_conversations():
    try:
        # Obtener todas las conversaciones de la colección
        conversations = list(conversations_collection.find({}, {"_id": 0}))
        
        # Devolver las conversaciones
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
connections = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            for conn in connections:
                await conn.send_text(data)
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
                response = await client.post("http:localhost:8000/embed", files=files)

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
    base_url: str  # Este campo es importante para las paginas que solamente son sistemas.php por poner un ejemplo

# Ruta para scrapear y generar PDF
@app.post("/scrape_and_embed")
async def scrape_and_embed(request: LinkRequest):
    links = request.links
    base_url = request.base_url  # Ahora esto debería funcionar correctamente
    print(f"Received links: {links}")
    print(f"Base URL: {base_url}")
    results = []
    
    for link in links:
        try:
            # Construir la URL completa si es relativa
            if link.startswith("http"):
                url = link  # Si el enlace ya es absoluto, úsalo directamente
            else:
                # Si el enlace es relativo, construye la URL completa
                url = f"{base_url.rstrip('/')}/{link.lstrip('/')}"

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
