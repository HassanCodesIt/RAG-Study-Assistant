import os
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from groq import Groq
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables once
load_dotenv()

# Initialize global instances (Singleton pattern)
chroma_client = chromadb.PersistentClient(path="./vecDB1")
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



def extraction(file_path):
    text = extract_text(file_path)
    return text





import re


def get_collection_name(subject: str) -> str:
    """
    Sanitizes and normalizes subject to satisfy ChromaDB collection name rules:
    - 3-512 characters
    - only [a-zA-Z0-9._-]
    - starts and ends with [a-zA-Z0-9]
    """
    clean = re.sub(r'[^a-zA-Z0-9._-]+', '_', (subject or "").strip().lower()).strip('._-')
    if not clean:
        clean = "default"
    if len(clean) < 3:
        clean = f"sub_{clean}"
    clean = clean.strip('._-')
    if len(clean) < 3:
        clean = f"sub_{clean}"
    return clean[:512]


def vectordbadd(text, subject):
    # Use global client
    collection_name = get_collection_name(subject)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)

    # Use global embedder
    embeddings = embedder.embed_documents(chunks)

    existing = len(collection.get()["ids"])
    ids = [f"id{existing + i + 1}" for i in range(len(chunks))]

    collection.add(
        embeddings=embeddings, 
        documents=chunks,
        ids=ids
    )

    return ids


def vectordbget(subject, query, top_k=8):
    # Use global client
    collection_name = get_collection_name(subject)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    # Retrieve up to top_k chunks, bounded by available chunks in the collection
    n_results = min(top_k, total_chunks)

    # Use global embedder
    query_embedding = embedder.embed_query(query)

    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    if results and results.get("documents") and len(results["documents"]) > 0:
        return results["documents"][0]
    return []


def llm(prompt, context):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "Error: GROQ_API_KEY is not set. Please ensure GROQ_API_KEY is defined in your .env file."

    if not context or (isinstance(context, list) and len(context) == 0):
        return "No relevant study materials found for this subject. Please make sure you have uploaded the PDF for this subject first."

    # Format context cleanly from list of chunks
    if isinstance(context, list):
        context_text = "\n\n---\n\n".join(chunk.strip() for chunk in context if chunk)
    else:
        context_text = str(context)

    client = Groq(api_key=api_key)

    content = f"""Answer the following question using the context provided below.

Context:
{context_text}

Question: {prompt}

Instructions:
1. Carefully review all parts of the context.
2. If the user asks about projects, skills, experience, or details present in the context, list and explain each one thoroughly.
3. Only state that something is not found if it is genuinely absent from the entire context.
4. Provide a clear, detailed, and well-structured answer.

Answer:"""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": content}],
            temperature=0.7,
            max_completion_tokens=2048,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
        )

        full_answer = ""
        for chunk in completion:
            full_answer += chunk.choices[0].delta.content or ""
        return full_answer
    except Exception as e:
        err_msg = str(e)
        try:
            fallback_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": content}],
                temperature=0.6,
                max_completion_tokens=2048,
                stream=True,
            )
            full_answer = ""
            for chunk in fallback_completion:
                full_answer += chunk.choices[0].delta.content or ""
            return full_answer
        except Exception as fallback_err:
            return f"Error from Groq API: {err_msg} (Fallback failed: {str(fallback_err)})"


