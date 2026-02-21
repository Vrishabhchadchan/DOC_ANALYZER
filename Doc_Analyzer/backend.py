import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

def create_rag_chain(pdf_path):
    """
    Takes a PDF path, processes it, and returns a RAG chain 
    capable of answering questions about that specific PDF.
    """
    
    # --- 1. Document Loading & Splitting ---
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split the document into smaller chunks (1000 characters)
    # This is crucial so the LLM doesn't get overwhelmed with too much text at once.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # --- 2. Vector Store (The "Brain") ---
    # We use an in-memory vector store (no persist_directory) 
    # so it resets automatically for every new file.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    # Create a retriever that looks for the 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # --- 3. LLM Setup ---
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct", # You can swap this for LLaMA 3 if you have access
        huggingfacehub_api_token="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",
        task="text-generation",
        temperature=0.7,
        max_new_tokens=512,
        do_sample=True
    )
    
    chat_model = ChatHuggingFace(llm=llm)

    # --- 4. RAG Chain Construction ---
    # --- 4. RAG Chain Construction ---
    system_prompt = (
        "You are an expert AI document analyzer. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, clearly state that you don't know. "
        "\n\n"
        "CRITICAL FORMATTING INSTRUCTIONS:\n"
        "1. You MUST format your entire response using proper Markdown.\n"
        "2. Use headings (## or ###) to organize different sections.\n"
        "3. Use standard Markdown bullet points (* or -) instead of inline symbols.\n"
        "4. You MUST add a blank new line before and after every bullet point or paragraph.\n"
        "5. Bold important keywords to make it scannable.\n"
        "\n\nContext:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain