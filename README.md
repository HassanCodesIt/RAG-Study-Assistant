

# 📘 PDF-Based Retrieval-Augmented Generation (RAG) System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that lets users upload PDFs (classified by subject), store their text embeddings in a **vector database**, and later **ask contextual questions** to retrieve precise, referenced answers from the document itself.

Built with **FastAPI**, **Groq LLM**, **HuggingFace Embeddings**, and **ChromaDB**, this system functions as a mini intelligent document Q&A platform.

---

## 🌟 Key Features

✅ **Smart PDF Uploading** – Upload PDFs categorized by subject.  
✅ **Automatic Text Extraction** – Uses `pdfminer` to extract readable text.  
✅ **Text Chunking** – Splits documents intelligently for better vectorization.  
✅ **Semantic Embeddings** – Encodes document chunks using HuggingFace’s `all-MiniLM-L6-v2`.  
✅ **Vector Database** – Stores and retrieves embeddings using `ChromaDB`.  
✅ **Contextual LLM Querying** – Uses `Groq`’s LLM API to generate elaborate, referenced answers.  
✅ **FastAPI Backend** – Manages upload and query routes efficiently.  
✅ **Frontend Interface** – Clean, responsive HTML pages for Uploading and Querying.  
✅ **Persistent Storage** – Uses `localStorage` to remember subjects locally.  
✅ **Progress Bars & Spinners** – Real-time user feedback during uploads.

---

## 🧠 Tech Stack

| Category | Technology | Purpose |
|-----------|-------------|----------|
| **Framework** | [FastAPI](https://fastapi.tiangolo.com/) | Backend API |
| **Language** | Python 3.10+ | Core application logic |
| **Embeddings Model** | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Vector representation |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Document store and retrieval |
| **PDF Processing** | [pdfminer.six](https://pypi.org/project/pdfminer.six/) | Text extraction from PDFs |
| **LLM API** | [Groq API](https://groq.com/) | Natural language question-answering |
| **Frontend** | HTML, CSS, Vanilla JS | User Interface |
| **Text Splitter** | [LangChain RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/) | Efficient chunking for long texts |

---

## ⚙️ Architecture Overview

### 🧩 Workflow Summary

1. **Upload Phase**
   - User uploads a PDF → FastAPI saves it → Text extracted using `pdfminer`.
   - Text is split into semantic chunks (1000 characters with 150 overlap).
   - Each chunk is embedded using HuggingFace model → Stored in ChromaDB.

2. **Query Phase**
   - User selects a subject and asks a question.
   - Query is embedded → Compared to stored vectors.
   - Top `k` similar chunks retrieved.
   - Chunks are passed as “context” to Groq LLM → Generates final answer.

---

## 🧭 System Architecture (Mermaid Diagram)

```mermaid
flowchart TD
A[User Uploads PDF] -->|POST /upload| B[FastAPI Backend]
B --> C[pdfminer: Extract Text]
C --> D[LangChain Splitter: Chunk Text]
D --> E[HuggingFace Embeddings]
E --> F[ChromaDB: Store Embeddings]

subgraph Database Layer
F[ChromaDB Vector Store]
end

G[User Asks Question] -->|POST /query| B
B --> H[HuggingFace Embedding for Query]
H --> I[ChromaDB: Similarity Search]
I --> J[Top-k Relevant Chunks]
J --> K[Groq LLM: Generate Answer]
K --> L[Response Sent to Frontend]
L --> M[Frontend Displays Answer]
````

---

## 🔁 Query Sequence (Mermaid Sequence Diagram)

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant FastAPI
    participant ChromaDB
    participant GroqLLM

    User->>Frontend: Select Subject + Type Question
    Frontend->>FastAPI: POST /query {subject, user_query}
    FastAPI->>ChromaDB: Query Vector Similarity
    ChromaDB-->>FastAPI: Return Top-k Relevant Chunks
    FastAPI->>GroqLLM: Send Context + Question
    GroqLLM-->>FastAPI: Return Generated Answer
    FastAPI-->>Frontend: JSON Response with Answer
    Frontend-->>User: Display Detailed Answer
```

---

## 🧩 Directory Structure

```
📦 project-root/
├── main.py                    # FastAPI backend app
├── subjects.py                # Core logic: PDF extraction, embedding, querying
├── templates/
│   ├── index.html             # Upload interface
│   └── query.html             # Query interface
├── savepdf/                   # Saved PDFs
├── vecDB1/                    # ChromaDB persistent store
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/pdf-rag-system.git
cd pdf-rag-system
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

#### requirements.txt

```text
fastapi
uvicorn
pdfminer.six
chromadb
langchain-text-splitters
langchain-community
sentence-transformers
groq
```

### 4️⃣ Set API Key

In `subjects.py`, replace:

```python
client = Groq(api_key="api_key")
```

with your actual Groq API key:

```python
client = Groq(api_key="YOUR_GROQ_API_KEY")
```

### 5️⃣ Run the Server

```bash
uvicorn main:app --reload
```

Open your browser and visit:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🌐 API Endpoints

| Method | Endpoint  | Description                                    |
| ------ | --------- | ---------------------------------------------- |
| `GET`  | `/`       | Returns Upload Page                            |
| `POST` | `/upload` | Uploads PDF, extracts text, creates embeddings |
| `GET`  | `/query`  | Returns Query Page                             |
| `POST` | `/query`  | Queries the LLM using Chroma context           |

---

## 🧠 Example Usage

**Step 1:** Upload

* Go to `/`
* Choose subject (e.g., *Chemistry*)
* Upload `organic_chemistry.pdf`

**Step 2:** Query

* Go to `/query`
* Choose *Chemistry*
* Ask: “Explain SN1 reaction mechanism”

**Step 3:** Output
LLM responds with:

```
The SN1 reaction mechanism proceeds via the formation of a carbocation...
(Mentioned in Paragraph 4 of the document)
```

---

## 💻 Frontend Overview

### Upload Page

* Add new or choose existing subject.
* Upload PDF with real-time progress.
* Success and error handling via inline messages.

### Query Page

* Select a subject.
* Enter a natural language query.
* Displays formatted answer in a response box.

---

## 🧩 Key Functions Explained

### `extraction(file_path)`

Extracts text content from PDF using `pdfminer.six`.

### `vectordbadd(text, subject)`

* Splits text into chunks.
* Embeds chunks using `HuggingFaceEmbeddings`.
* Stores embeddings + text into ChromaDB.

### `vectordbget(subject, query)`

* Embeds query.
* Retrieves top-k relevant document chunks.

### `llm(prompt, context)`

* Sends question and retrieved context to Groq model.
* Returns an elaborate, referenced, markdown-free answer.

---

## 🧰 Local Storage Functionality

Frontend uses `localStorage` to persist subjects:

* New subjects added dynamically.
* Survive page refreshes.
* Makes switching between topics seamless.

---

## ⚡ Performance Considerations

* Efficient chunk size (1000 chars, 150 overlap).
* Fast retrieval via ChromaDB vector search.
* Lightweight transformer model ideal for CPU inference.
* Streaming response handling for low latency.

---

## 🧱 Future Enhancements

* [ ] Multi-file management per subject
* [ ] Embedding caching for faster reloads
* [ ] Authentication & user profiles
* [ ] Integration with local models (DeepSeek, Ollama)
* [ ] UI improvements with chat-like experience

---

## 📊 Architecture Diagram (Rendered via Mermaid)

(See interactive diagram above in FigJam preview)

---

## 🧾 License

MIT License © 2025 [Your Name]
You are free to use, modify, and distribute this project.

---

## 🙏 Acknowledgments

* [LangChain](https://www.langchain.com/)
* [HuggingFace](https://huggingface.co/)
* [Groq](https://groq.com/)
* [ChromaDB](https://www.trychroma.com/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [pdfminer.six](https://pypi.org/project/pdfminer.six/)

---

### 🏁 Summary

This project is a **complete end-to-end RAG (Retrieval-Augmented Generation)** application — combining LLM intelligence with semantic retrieval and a clean user interface.
It showcases practical integration of **document parsing**, **vector search**, and **language model reasoning** in a modular and extensible architecture.

---

✨ *Developed with Python, FastAPI, Groq LLM, and a lot of ☕ caffeine.*






```
