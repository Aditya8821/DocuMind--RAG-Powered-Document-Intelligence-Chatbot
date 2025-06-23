# DocuMind: RAG-Powered PDF Chatbot

A Retrieval-Augmented Generation (RAG) powered chatbot that allows you to chat with your PDF documents. Upload PDFs and ask questions about their content.

## Features

- **PDF Upload**: Upload any PDF document (resumes, policy docs, reports, etc.)
- **RAG Architecture**: Uses state-of-the-art RAG approach for accurate answers
- **Vector Database**: Stores document embeddings for efficient retrieval
- **Interactive Chat Interface**: User-friendly chat interface powered by Streamlit
- **Adaptive RAG Trigger (RAGate)**: Smart retrieval decisions to optimize performance
- **Document Comparison**: Compare multiple documents with comparative queries (e.g., "Who is better at web development?" when comparing resumes)

## Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python with LangChain for orchestration
- **LLM**: Google's Gemini API
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)

## Setup Instructions

### Prerequisites

- Python 3.8+ installed
- Google Gemini API key ([Get one here](https://makersuite.google.com/))

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/documind.git
   cd documind
   ```

2. Install dependencies:
   ```
   pip install -r backend/requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file in the backend directory
   - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

### Running the Application

Run the Streamlit app:
```
streamlit run app.py
```

## Usage

1. Upload PDF documents using the sidebar upload button
2. Wait for the documents to be processed and added to the vector database
3. Ask questions in the chat input field
4. View answers generated based on the content of your PDFs
5. For document comparison, upload multiple documents (e.g., resumes, reports) and ask comparative questions like "Who has more experience in web development?" or "Which document discusses sustainability more thoroughly?"

## How It Works

1. **Document Processing**: PDFs are uploaded, text is extracted, and split into chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Retrieval**: When a question is asked, the system retrieves the most relevant chunks
4. **Generation**: The LLM generates an answer based on the retrieved context

## Technical Details

This project demonstrates core RAG concepts:

- **Document Chunking**: Breaking large documents into manageable pieces
- **Vector Embeddings**: Converting text to numerical representations
- **Semantic Search**: Finding relevant context based on meaning, not just keywords
- **Context-Augmented Generation**: Providing the LLM with relevant context for accurate answers
- **Adaptive RAG Trigger (RAGate)**: Intelligently deciding when to use retrieval based on query type
- **Cross-Document Analysis**: Analyzing and comparing information across multiple documents to answer comparative questions about their content

## Advanced RAGate Implementation

The project includes an innovative RAGate system that adaptively decides when to use retrieval:

- **Pattern Recognition**: Identifies query types that likely need document knowledge
- **Confidence Scoring**: Calculates how likely a query requires document context
- **Thresholding**: Customizable confidence threshold for retrieval decisions
- **Debug Mode**: Optional visualization of retrieval decisions
- **Performance Optimization**: Reduces unnecessary retrievals for general questions
