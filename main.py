import os
import base64
import uuid
from base64 import b64decode
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_community.llms import HuggingFaceHub
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf

# Instalación de paquetes necesarios (comentar si ya están instalados)
# os.system("pip install -Uq unstructured[all-docs] pillow lxml pillow")
# os.system("pip install -Uq chromadb tiktoken langchain langchain-community langchain-openai langchain-groq")
# os.system("pip install -Uq python_dotenv langchain-huggingface custom_st langchain-community huggingface_hub")

# Configuración de variables de entorno
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Extracción de datos del PDF
output_path = "./finalbot/docs/"
file_path = output_path + 'Respuestas_basicas.pdf'

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

# Separar elementos extraídos en tablas, texto e imágenes
tables = []
texts = []
for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)
    if "CompositeElement" in str(type(chunk)):
        texts.append(chunk)

def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images = get_images_base64(chunks)

def display_base64_image(base64_code):
    image_data = base64.b64decode(base64_code)
    with open("output_image.jpg", "wb") as img_file:
        img_file.write(image_data)
    print("Image saved as output_image.jpg")

# Generación de resúmenes
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.
Respond only with the summary, no additional comment.
Table or text chunk: {element}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)
model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
tables_html = [table.metadata.text_as_html for table in tables]
table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

# Creación del vectorstore
vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)
store = InMemoryStore()
id_key = "doc_id"
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

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
    for image in docs_by_type["images"]:
        prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature": 0.5, "max_length": 512})

chain = ({
    "context": retriever | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
} | RunnableLambda(build_prompt) | llm | StrOutputParser())

chain_with_sources = {
    "context": retriever | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(response=(RunnableLambda(build_prompt) | llm | StrOutputParser()))

response = chain.invoke("certificados?")
print(response)

response = chain_with_sources.invoke("Cuales son los certificados?")
print("Response:", response['response'])
for text in response['context']['texts']:
    print(text.text)
    print("Page number:", text.metadata.page_number)
    print("\n" + "-"*50 + "\n")
for image in response['context']['images']:
    display_base64_image(image)
