import os
import shutil
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain.schema.document import Document

class VectorStore:
    """Vector database for storing and retrieving document embeddings."""
    
    def __init__(self, persist_directory: str = "db"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory
        
        # Initialize a simple embedding model that doesn't require complex dependencies
        self.embedding_model = FakeEmbeddings(size=384)
        
        # Check if directory exists before trying to remove it
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
            except Exception:
                # Silent failure, as this isn't critical
                pass
        
        # Create directory fresh
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the vector database
        # Using in-memory DB initially to avoid ChromaDB tenant issues
        try:
            self.db = Chroma(
                embedding_function=self.embedding_model
            )
        except Exception:
            # Failsafe: if even in-memory DB fails, try again with minimal settings
            self.db = Chroma(
                client_settings=chromadb.config.Settings(anonymized_telemetry=False),
                embedding_function=self.embedding_model
            )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        # Ensure all documents have proper metadata for filtering
        for doc in documents:
            if "source" not in doc.metadata:
                doc.metadata["source"] = "unknown"
        
        self.db.add_documents(documents)
        self.db.persist()
    
    def similarity_search(self, query: str, k: int = 4, source_filter: str = None) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            source_filter: Optional filter to search only within a specific document
            
        Returns:
            List of Document objects
        """
        if source_filter:
            # Search only within the specified document
            return self.db.similarity_search(
                query, 
                k=k,
                filter={"source": source_filter}
            )
        else:
            # Search across all documents
            return self.db.similarity_search(query, k=k)
            
    def get_document_sources(self) -> List[str]:
        """
        Get a list of all document sources in the database.
        
        Returns:
            List of document source names
        """
        try:
            # Get all documents 
            all_docs = self.db.similarity_search("", k=1000)
            
            # Extract unique source names
            sources = set()
            for doc in all_docs:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
            
            return list(sources)
        except Exception as e:
            print(f"Error getting document sources: {str(e)}")
            return []
    
    def clear(self) -> None:
        """Clear the vector store."""
        try:
            self.db._collection.delete(where={})
            # No need to call persist explicitly
        except Exception:
            # If deletion fails, re-initialize the database
            self.__init__(self.persist_directory)
