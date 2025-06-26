import os
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema.document import Document

class FAISSVectorStore:
    """
    A FAISS-based vector store implementation that provides efficient similarity search
    and document retrieval using dense vector embeddings.
    
    This implementation uses:
    - Sentence Transformers: For generating high-quality document embeddings that capture semantic meaning
    - FAISS (Facebook AI Similarity Search): For efficient similarity search and vector indexing
    
    Key Features:
    - Dense vector embeddings for semantic understanding
    - Fast and efficient similarity search using FAISS
    - L2 distance metric for measuring document similarity
    - Support for document source filtering
    - Proper metadata handling for documents
    
    The default model (all-MiniLM-L6-v2) provides a good balance between:
    - Performance: Fast encoding and similarity search
    - Quality: Good semantic understanding
    - Size: Relatively small model that works well for most use cases
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the FAISS vector store.
        
        Args:
            model_name: Name of the sentence transformer model to use for embeddings.
                      Defaults to 'all-MiniLM-L6-v2' which provides a good balance
                      between performance and quality.
        
        The initialization process:
        1. Loads the specified sentence transformer model
        2. Gets the embedding dimension from the model
        3. Initializes a FAISS index using L2 distance metric
        4. Sets up storage for documents and their metadata
        """
        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name)
        
        # Initialize empty FAISS index
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Store documents and their metadata
        self.documents = []
        self.document_sources = set()
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate a dense vector embedding for a given text string.
        
        This method uses the sentence transformer model to convert text into
        a fixed-size vector that captures its semantic meaning. Similar texts
        will have similar vector representations.
        
        Args:
            text: Text string to convert into an embedding vector
            
        Returns:
            Numpy array containing the embedding vector
        """
        return self.model.encode([text])[0]
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store by converting them to embeddings.
        
        This method:
        1. Generates embeddings for all documents in a batch
        2. Adds the embeddings to the FAISS index
        3. Stores the original documents and their metadata
        4. Updates the set of document sources
        
        Args:
            documents: List of Document objects to add to the vector store
        """
        if not documents:
            return
            
        # Get embeddings for all documents
        embeddings = self.model.encode([doc.page_content for doc in documents])
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and update sources
        for doc in documents:
            # Ensure document has source metadata
            if "source" not in doc.metadata:
                doc.metadata["source"] = "unknown"
            
            # Add document source to our set of sources
            self.document_sources.add(doc.metadata["source"])
            
            # Add the document to our list
            self.documents.append(doc)
    
    def similarity_search(self, query: str, k: int = 4, source_filter: str = None) -> List[Document]:
        """
        Perform semantic similarity search using FAISS.
        
        This method:
        1. Converts the query into an embedding vector
        2. Uses FAISS to find the k nearest neighbors based on L2 distance
        3. Retrieves the corresponding documents
        4. Applies source filtering if specified
        
        The search process ensures semantic matching rather than just keyword matching,
        meaning it can find relevant documents even if they use different but related terms.
        
        Args:
            query: Query string to search for similar documents
            k: Number of similar documents to return (default: 4)
            source_filter: Optional filter to search only within a specific document source
            
        Returns:
            List of Document objects sorted by similarity to the query
        """
        if not self.documents:
            return []
            
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # If source filter is applied, we need to search through all results
        # and filter afterward to ensure we get enough matches
        search_k = k if not source_filter else len(self.documents)
        
        # Perform similarity search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            min(search_k, len(self.documents))
        )
        
        # Get matching documents
        matches = [self.documents[i] for i in indices[0]]
        
        # Apply source filter if specified
        if source_filter:
            matches = [doc for doc in matches if doc.metadata.get("source") == source_filter]
            matches = matches[:k]  # Limit to k results after filtering
        
        return matches
    
    def get_document_sources(self) -> List[str]:
        """
        Get a list of all document sources in the database.
        
        Returns:
            List of document source names
        """
        return list(self.document_sources)
    
    def clear(self) -> None:
        """Clear the vector store."""
        # Reset FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Clear documents and sources
        self.documents = []
        self.document_sources = set()
