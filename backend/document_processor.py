import os
from typing import List, Dict
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

class DocumentProcessor:
    """Class for processing PDF documents and chunking text."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        return text
    
    def chunk_text(self, text: str) -> List[Document]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of Document objects
        """
        return self.text_splitter.create_documents([text])
    
    def process_pdf(self, pdf_path: str, original_filename: str = None) -> List[Document]:
        """
        Process a PDF file: extract text and split into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            original_filename: Original filename to use as source (instead of temp filename)
            
        Returns:
            List of Document objects
        """
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        
        # Use original filename if provided, otherwise use the basename of the path
        source_name = original_filename if original_filename else os.path.basename(pdf_path)
        
        # Add source metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source": source_name,
                "chunk_id": i
            })
        
        return chunks
