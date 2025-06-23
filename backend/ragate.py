"""
RAGate: Adaptive RAG Trigger Module

This module implements a gating mechanism that intelligently decides when to use retrieval
in a Retrieval-Augmented Generation (RAG) system based on the type of query.
"""

import re
from typing import List, Dict, Any, Tuple, Optional

class RAGate:
    """
    Implements an adaptive triggering mechanism for RAG systems.
    
    The RAGate class analyzes input queries to determine whether retrieval is necessary,
    helping reduce latency and costs for queries that don't require document context.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the RAGate with configuration parameters.
        
        Args:
            confidence_threshold: Threshold for retrieval decision (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        
        # Patterns that likely require document knowledge
        self.document_patterns = [
            r"(in|within|from|according to|based on|referring to|mention(ed)? (in|by))(\s+the)?\s+(document|pdf|text|file)",
            r"what (does|do|did) .{3,30} (say|state|mention|explain|describe|mean|discuss)",
            r"where (in|within) .{3,30} (is|are|was|were) .{3,30} (mention|discuss|state|list|explain)",
            r"how (many|much) .{3,30} (mention|list|have|has|contain|include)",
            r"who (is|are|was|were) .{3,50} (according to|in|within|mention)",
            r"when (did|was|were|is|are) .{3,50} (according to|in|within|mention)",
            r"which (section|part|paragraph|page|chapter|area)",
            r"tell me about .{3,50} (in|from|within) .{3,50}",
            r"list .{3,50} (mention|discuss|state|describe)",
            r"summarize .{3,50} (section|part|chapter|document|pdf|text)",
            r"what (is|are) .{3,50} (about|regarding)",
            r"explain .{3,50} (concept|idea|topic|subject|theory)",
            r"describe .{3,50} (in|from|within|about)",
            r"extract .{3,50} (from|in)",
            r"find .{3,50} (in|within|from)",
        ]
        
        # Patterns that likely don't require document knowledge
        self.general_patterns = [
            r"^(hi|hello|hey|greetings|howdy)",
            r"^how (are|is|was|were) you",
            r"^what (are|is) your",
            r"(thank|thanks)",
            r"^(bye|goodbye|see you)",
            r"^(help|assist)",
            r"^who (created|made|built|developed) you",
            r"can you help",
            r"tell me about yourself",
            r"what can you do",
            r"how do (I|we) use",
            r"what (is|are) RAG",
            r"explain how (you|this) work",
        ]
        
        # Compile all patterns for efficiency
        self.document_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.document_patterns]
        self.general_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.general_patterns]
    
    def decide(self, query: str) -> Tuple[bool, float]:
        """
        Decide whether to use retrieval for the given query.
        
        Args:
            query: The user query to analyze
            
        Returns:
            Tuple of (use_retrieval: bool, confidence: float)
        """
        # Check document-specific patterns (high likelihood of needing retrieval)
        for pattern in self.document_regex:
            if pattern.search(query):
                # Calculate a confidence score (could be refined in a real implementation)
                return True, 0.9
        
        # Check general patterns (low likelihood of needing retrieval)
        for pattern in self.general_regex:
            if pattern.search(query):
                return False, 0.8
                
        # For queries that don't match any patterns, default behavior
        # This is a simplified heuristic:
        # - Longer queries are more likely to need document context
        # - Queries with question words are more likely to need document context
        
        # Check for question words
        question_words = ["what", "who", "where", "when", "why", "how", "which", "can", "does", "do"]
        has_question_word = any(word in query.lower().split() for word in question_words)
        
        # Length-based heuristic
        query_length = len(query.split())
        length_factor = min(query_length / 20.0, 1.0)  # Normalize by typical question length, The min() ensures that once a query is "long enough" (20+ words), additional length doesn't artificially inflate confidence beyond the meaningful 0-1 range.
        
        # Calculate confidence score
        # Higher for longer queries with question words
        confidence = 0.5 + (0.3 if has_question_word else 0) + (0.2 * length_factor)
        
        # Decision based on confidence threshold
        return confidence >= self.confidence_threshold, confidence
    
    def explain_decision(self, query: str) -> str:
        """
        Provide an explanation for the retrieval decision.
        
        Args:
            query: The user query
            
        Returns:
            Explanation string
        """
        use_retrieval, confidence = self.decide(query)
        
        if use_retrieval:
            if confidence > 0.8:
                return f"Using retrieval (confidence: {confidence:.2f}): Query strongly suggests document-specific information is needed."
            else:
                return f"Using retrieval (confidence: {confidence:.2f}): Query likely requires document context."
        else:
            return f"Skipping retrieval (confidence: {confidence:.2f}): Query can likely be answered without document context."
