import streamlit as st
import weaviate
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Streamlit title and description
st.title("RAG Question-Answering System")
st.write("Ask a question related to the provided PDF document and get AI-based answers.")

# Weaviate API setup
WEAVIATE_API_KEY = "79JNO58NLZcKLRyfPcPQzpz5AhGwK3FlqOJ9"
WEAVIATE_CLUSTER = "https://xppgynodsewowzqvoktmgw.c0.us-east1.gcp.weaviate.cloud"

client = weaviate.Client(
    url=WEAVIATE_CLUSTER, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)

# Initialize Hugging Face Embedding model
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
  model_name=embedding_model_name,
)

# Initialize the document loader
loader = PyPDFLoader("RAG.pdf", extract_images=True)
pages = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(pages)

# Initialize Weaviate Vector DB
vector_db = Weaviate.from_documents(docs, embeddings, client=client, by_text=False)

# Initialize Hugging Face Model
huggingfacehub_api_token = "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"
model = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 1, "max_length": 180}
)

# Prompt template
template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Output parser
output_parser = StrOutputParser()

# Retriever from Vector DB
retriever = vector_db.as_retriever()

# RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

# Streamlit input for the question
question = st.text_input("Ask a question:")

# Button to submit the question
if st.button("Get Answer"):
    if question:
        try:
            # Perform the RAG model query
            answer = rag_chain.invoke(question)
            st.write(f"**Answer**: {answer}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please provide a valid question.")
