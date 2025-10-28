import os
import bs4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

# 1. Load GROQ_API_KEY từ env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment")

# 2. Tải nội dung nguồn (fixed cứng 1 link, demo)
def load_documents():
    loader = WebBaseLoader(
        web_paths=("https://vi.wikipedia.org/wiki/B%C3%A3o",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(id=("mw-content-text"))
        ),
    )
    docs = loader.load()
    return docs

# 3. Chia nhỏ tài liệu thành chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
    )
    return splitter.split_documents(docs)

# 4. Tạo embeddings và vectorstore in-memory
def build_vectorstore(splits):
    hf = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    vectorstore = Chroma.from_documents(splits, embedding=hf)
    return vectorstore

# 5. Tạo LLM Groq
def build_llm():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.5,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )
    return llm

# 6. Build RAG chain (retrieval -> prompt -> llm -> parse)
def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
{context}

Question: {question}
"""
    )

    rag_chain = (
        {
            "context": retriever,           # tự động lấy context theo question
            "question": RunnablePassthrough()  # chính câu hỏi
        }
        | prompt
        | llm
        | StrOutputParser()               # trả về chuỗi text thuần
    )

    return rag_chain

# 7. Hàm khởi tạo toàn bộ pipeline (gọi lúc server start)
def init_pipeline():
    docs = load_documents()
    splits = split_documents(docs)
    vectorstore = build_vectorstore(splits)
    llm = build_llm()
    rag_chain = build_rag_chain(vectorstore, llm)
    return rag_chain
