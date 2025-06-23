import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.schema.runnable import RunnablePassthrough
from backend.ragate import RAGate

# Load environment variables
load_dotenv()

class RAGChatbot:
    """RAG-powered chatbot for answering questions about PDF documents."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-002", 
                 confidence_threshold: float = 0.7,
                 use_ragate: bool = True):
        """
        Initialize the RAG chatbot.
        
        Args:
            model_name: Name of the Gemini model to use
            confidence_threshold: Threshold for RAGate retrieval decision
            use_ragate: Whether to use the RAGate system for adaptive retrieval
        """
        # Check if API key is available
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in the .env file.")
        
        # Initialize the RAGate system
        self.ragate = RAGate(confidence_threshold=confidence_threshold)
        self.use_ragate = use_ragate
        
        # Initialize the language model
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.3,
            max_output_tokens=2048,
        )
        
        # Define the QA prompt template with document context
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an intelligent assistant that answers questions based on provided context.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {question}
            
            INSTRUCTIONS:
            1. Answer the question based only on the provided context.
            2. If the context doesn't contain enough information to answer the question, just say "I don't have enough information to answer that question."
            3. Provide a detailed and informative answer.
            4. Format your answer in a clear and readable way.
            5. If appropriate, use bullet points or numbered lists.
            
            ANSWER:
            """
        )
        
        # Define the direct answer prompt template (no document context)
        self.direct_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an intelligent assistant that answers general questions and helps users with their PDF chatbot.
            
            QUESTION:
            {question}
            
            INSTRUCTIONS:
            1. Answer the question directly and concisely.
            2. If the question is about specific document content, explain that you need to access the document.
            3. Provide helpful information about general concepts related to RAG, chatbots, or PDFs if relevant.
            4. Format your answer in a clear and readable way.
            
            ANSWER:
            """
        )
        
        # Initialize the QA chain using the modern RunnableSequence approach
        self.qa_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.qa_prompt
            | self.llm
        )
        
        # Initialize the direct answer chain (no retrieval)
        self.direct_chain = (
            {"question": RunnablePassthrough()}
            | self.direct_prompt
            | self.llm
        )
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format a list of documents into a context string.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "Unknown source")
            context_parts.append(f"Document {i+1} (from {source}):\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def decide_retrieval(self, question: str) -> Tuple[bool, float, str]:
        """
        Decide whether to use retrieval for this question.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (use_retrieval, confidence, explanation)
        """
        # If RAGate is disabled, always use retrieval
        if not self.use_ragate:
            return True, 1.0, "RAGate disabled, using retrieval for all questions"
        
        # Use RAGate to decide
        use_retrieval, confidence = self.ragate.decide(question)
        explanation = self.ragate.explain_decision(question)
        
        return use_retrieval, confidence, explanation
    
    def direct_answer(self, question: str) -> str:
        """
        Generate a direct answer without using document retrieval.
        
        Args:
            question: Question to answer
            
        Returns:
            Direct answer (not using document context)
        """
        try:
            response = self.direct_chain.invoke({
                "question": question
            })
            return response.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def answer_with_retrieval(self, question: str, documents: List[Document]) -> str:
        """
        Answer a question using retrieved document context.
        
        Args:
            question: Question to answer
            documents: List of Document objects to use as context
            
        Returns:
            Answer based on document context
        """
        if not documents:
            return "I don't have any documents to reference for answering your question."
        
        context = self.format_context(documents)
        
        try:
            response = self.qa_chain.invoke({
                "context": context,
                "question": question
            })
            return response.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def answer_question(self, question: str, documents: List[Document]) -> str:
        """
        Answer a question, adaptively using retrieval based on the question type.
        
        Args:
            question: Question to answer
            documents: List of Document objects to use as context (if needed)
            
        Returns:
            Answer to the question
        """
        # Decide whether to use retrieval
        use_retrieval, confidence, explanation = self.decide_retrieval(question)
        
        # Debug information (could be logged or returned to the user in a debug mode)
        debug_info = f"\n\nRAGate: {explanation}" if self.use_ragate else ""
        
        # Answer based on decision
        if use_retrieval:
            answer = self.answer_with_retrieval(question, documents)
        else:
            answer = self.direct_answer(question)
        
        # For now, we'll return the answer without the debug info
        # But in a real application, you might want to include a debug mode
        # return answer + debug_info
        return answer
