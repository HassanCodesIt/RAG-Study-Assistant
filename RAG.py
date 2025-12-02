import os
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from groq import Groq
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings



def extraction(file_path):
    text = extract_text(file_path)
    return text





def vectordbadd(text, subject):
    
    chroma_client = chromadb.PersistentClient(path="./vecDB1")
    collection = chroma_client.get_or_create_collection(name=subject)

   
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)

   
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.embed_documents(chunks)

    
    

    
    existing = len(collection.get()["ids"])
    ids = [f"id{existing + i + 1}" for i in range(len(chunks))]

    
    collection.add(
        embeddings=embeddings, 
        documents=chunks,
        ids=ids
    )


    return ids

    




def vectordbget(subject, query, top_k=3):
    chroma_client = chromadb.PersistentClient(path="./vecDB1")
    collection = chroma_client.get_or_create_collection(name=subject)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embedder.embed_query(query)

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    documents = results["documents"][0]  
    return documents




def llm(prompt, context):
    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    content = f"Answer the following question using only the data below.\n\nContext:\n{context}\n\nQuestion: {prompt}. if the question isnt from the context of the data or only entirely different from the given data, answer something like out of context of the pdf or anything similar and don't   \n also make the answer detailed and elaborate \n also provide where it is mentioned like para 5 or line 6 etc in brackets a line below\n also avoid using markdown \n\nAnswer:"

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": content}],
        temperature=0.7,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
    )

    full_answer = ""
    for chunk in completion:full_answer += chunk.choices[0].delta.content or ""
    return full_answer



