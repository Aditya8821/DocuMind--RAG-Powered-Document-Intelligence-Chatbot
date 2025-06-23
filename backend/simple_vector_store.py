import os
from typing import List, Dict, Any
import numpy as np
from langchain.schema.document import Document

class SimpleVectorStore:
    """
    A simple in-memory vector store implementation that doesn't rely on ChromaDB.
    This is a basic implementation for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize the simple vector store."""
        self.documents = []
        self.document_sources = set()
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        # Add documents to our store
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
        Perform a basic search on documents.
        This is a simplified implementation that doesn't use actual embeddings.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            source_filter: Optional filter to search only within a specific document
            
        Returns:
            List of Document objects
        """
        # Filter documents by source if needed
        if source_filter:
            filtered_docs = [doc for doc in self.documents if doc.metadata.get("source") == source_filter]
            
            # If no documents match the filter, return empty list
            if not filtered_docs:
                return []
            
            # For a specific source, just return up to k documents
            return filtered_docs[:min(k, len(filtered_docs))]
        else:
            # For "All Documents" selection, ensure a balanced distribution
            # Group documents by source
            docs_by_source = {}
            for doc in self.documents:
                source = doc.metadata.get("source", "unknown")
                if source not in docs_by_source:
                    docs_by_source[source] = []
                docs_by_source[source].append(doc)
            
            # If no documents at all, return empty list
            if not docs_by_source:
                return []
            
            # Calculate how many documents to take from each source
            sources = list(docs_by_source.keys())
            num_sources = len(sources)
            
            if num_sources == 0:
                return []
            
            # If we have fewer sources than k, we'll take more from each source
            docs_per_source = max(1, k // num_sources)
            
            # Collect balanced results
            results = []
            for source in sources:
                source_docs = docs_by_source[source]
                results.extend(source_docs[:min(docs_per_source, len(source_docs))])
                
                # If we have enough results, stop
                if len(results) >= k:
                    break
            
            # If we still need more documents to reach k, add more from the remaining sources
            if len(results) < k:
                remaining = k - len(results)
                for source in sources:
                    source_docs = docs_by_source[source]
                    # Skip the docs we've already added
                    start_idx = min(docs_per_source, len(source_docs))
                    additional_docs = source_docs[start_idx:start_idx + remaining]
                    results.extend(additional_docs)
                    remaining -= len(additional_docs)
                    if remaining <= 0:
                        break
            
            # Limit to k results
            return results[:k]
    
    def get_document_sources(self) -> List[str]:
        """
        Get a list of all document sources in the database.
        
        Returns:
            List of document source names
        """
        return list(self.document_sources)
    
    def clear(self) -> None:
        """Clear the vector store."""
        self.documents = []
        self.document_sources = set()
