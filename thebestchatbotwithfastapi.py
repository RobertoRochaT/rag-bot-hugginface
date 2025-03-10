from fastapi import FastAPI, HTTPException, UploadFile, File
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

# Configuración de la carpeta temporal
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Configuración de MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
db = client.chatbotlaguna
conversations_collection = db.conversations
documents_collection = db.documents

# Configuración del vectorstore
vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
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



# Función para parsear documentos
def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}


# Función para construir el prompt
def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = "".join(text_element.text for text_element in docs_by_type["texts"])

    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and images.
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

@app.get("/")
async def read_root():
    return {"message": "Bienvenido al Chatbot Laguna"}

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


# Ruta para realizar consultas
@app.post("/query")
async def query(request: QueryRequest):
    try:
        response = chain.invoke(request.question)

        # Guardar la conversación en MongoDB
        conversation_data = {
            "conversation_name": request.conversation_name,
            "question": request.question,
            "response": response,
            "timestamp": datetime.now()
        }
        conversations_collection.insert_one(conversation_data)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# Crear la carpeta de documentos si no existe
if not os.path.exists("./documents"):
    os.makedirs("./documents")

# Iniciar la aplicación
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
