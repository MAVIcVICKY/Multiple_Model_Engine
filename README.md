# 🌌 Multimodal Image Search Engine

Hey there! This is a project I built to explore the power of **Multimodal AI**. It's a search engine that doesn't just look for keywords—it actually "understands" the content of images and text. You can search for photos using natural language or even upload an image to find visually similar ones.

Technically, it bridges the gap between vision and language using **Google Gemini's** embedding models and **Pinecone's** vector storage.

---

## 🛠️ The Tech Behind It
- **Backend**: Django & Django REST Framework (The foundation)
- **AI Brain**: Google Gemini (`gemini-embedding-2-preview`) — used for generating 3072-dimensional multimodal embeddings.
- **Vector Database**: Pinecone — for lightning-fast similarity searching using Cosine Similarity.
- **Frontend**: A clean, dark-themed UI with real-time vector visualization.

---

## 🔄 How It Works (The Workflow)

I've broken the project down into three main stages. Here’s what happens under the hood:

### 1. Ingestion Stage (Building the Memory)
I used the **Flickr8k dataset** to populate the system.
*   **The Process**: I wrote a custom Django management command that iterates through images and their captions.
*   **The Transformation**: Each image is sent to Gemini to get an "Image Vector," and its caption is sent to get a "Text Vector."
*   **Storage**: Both vectors are stored in Pinecone. Because they are in the same 3072-dim space, the system knows that a "photo of a dog" and the text "golden retriever" are mathematically close to each other.

### 2. Retrieval Stage (The Search)
When you type something like *"a sunset at the beach"*:
*   **Step A**: The system turns your text into a query vector using Gemini.
*   **Step B**: It performs a **K-Nearest Neighbors (KNN)** search in Pinecone.
*   **Step C**: Pinecone calculates the **Cosine Similarity** and returns the Top-6 most relevant images.

### 3. Frontend Experience
*   **Vector Visualizer**: I added a visualizer that shows the first 10 dimensions of the embedding. It’s a cool way to see how the AI "perceives" your query.
*   **Image Search**: You can upload an image, and the system will find other images in the database that have a similar visual "signature."

---

## 🚀 Setting Up Locally

If you want to run this on your machine:

1.  **Clone it**: `git clone <repo-url>`
2.  **Environment**: Create a `.env` file (see `.env.example`). You'll need API keys for Gemini and Pinecone.
3.  **Install dependencies**: `pip install -r requirements.txt`
4.  **Ingest some data**:
    ```bash
    # This will ingest 100 images to get you started
    python manage.py ingest_data --limit 100
    ```
5.  **Run it**: `python manage.py runserver`

---

## 📊 My Management Commands
I added a few helper commands to make life easier:
*   `python manage.py ingest_data`: To push files to the cloud.
*   `python manage.py pinecone_status`: To quickly check how many vectors are live in the database.

---


## 🏗️ Complete System Flow

```mermaid
flowchart LR
    A[User] --> B{Input Type}

    B -->|Text Query| C[Text Embedding]
    B -->|Image Upload| D[Image Embedding]

    C --> E[Unified Vector Space]
    D --> E

    subgraph Ingestion Pipeline
        F[Flickr8k Dataset] --> G[Images + Captions]
        G --> H[Embedding Generation]
        H --> I[Vector Storage]
    end

    I --> J[Pinecone DB]

    E --> J

    J --> K[Similarity Search]
    K --> L[Top Results]

    L --> M[Django Backend]
    M --> N[Frontend UI]
