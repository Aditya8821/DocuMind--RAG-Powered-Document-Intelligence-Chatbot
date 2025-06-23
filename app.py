import os
import tempfile
import streamlit as st
from backend.document_processor import DocumentProcessor
from backend.simple_vector_store import SimpleVectorStore
from backend.rag_chatbot import RAGChatbot
from backend.ragate import RAGate

# Page configuration
st.set_page_config(
    page_title="DocuMind - RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

# Add custom CSS for better visual appearance
st.markdown("""
<style>
    /* Main styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(90deg, #4F46E5, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* User inputs and selectbox styling */
    .stSelectbox > div[data-baseweb="select"] > div {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #4F46E5;
    }
    
    /* Chat input container */
    .stChatInputContainer {
        padding: 0.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Divider styling */
    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = SimpleVectorStore()
if "document_processor" not in st.session_state:
    st.session_state.document_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []
if "use_ragate" not in st.session_state:
    st.session_state.use_ragate = True
if "show_debug_info" not in st.session_state:
    st.session_state.show_debug_info = False
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.7

# Application header with improved styling and concise description - made smaller
st.markdown("""
<div style="text-align: center; padding: 1rem; margin-bottom: 1.5rem; background: linear-gradient(90deg, rgba(79, 70, 229, 0.1), rgba(59, 130, 246, 0.1)); border-radius: 10px; max-width: 800px; margin-left: auto; margin-right: auto;">
    <h1 style="font-size: 2.5rem; margin-bottom: 0.3rem;">📄 DocuMind</h1>
    <p style="font-size: 1rem; margin-top: 0; color: #6B7280;">Chat intelligently with your PDF documents</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem; color: #6B7280; margin-bottom: 0;">
        RAG-powered chatbot for intelligent document Q&A
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for document upload and settings with improved styling
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h3 style="color: #4F46E5; font-weight: 600;">📤 Upload Documents</h3>
    </div>
    """, unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload your PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        all_processed = True
        for pdf_file in uploaded_files:
            if pdf_file.name not in st.session_state.loaded_files:
                st.text(f"Processing: {pdf_file.name}")
                
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Process the PDF file using the original filename as source
                    documents = st.session_state.document_processor.process_pdf(
                        pdf_path=tmp_path,
                        original_filename=pdf_file.name
                    )
                    
                    # Add documents to vector store
                    st.session_state.vector_store.add_documents(documents)
                    
                    # Force update document sources after adding new documents
                    if "document_sources" in st.session_state:
                        del st.session_state["document_sources"]
                    
                    # Add file to loaded files list
                    st.session_state.loaded_files.append(pdf_file.name)
                    
                    st.success(f"✅ {pdf_file.name} processed and added to database!")
                except Exception as e:
                    st.error(f"❌ Error processing {pdf_file.name}: {str(e)}")
                    all_processed = False
                finally:
                    # Clean up the temporary file
                    os.unlink(tmp_path)
        
        if all_processed and st.session_state.loaded_files:
            st.success(f"All documents processed! You can now ask questions about them.")
    
    st.divider()
    
    # Add option to clear the database
    if st.button("🗑️ Clear Database"):
        st.session_state.vector_store.clear()
        st.session_state.loaded_files = []
        st.session_state.chat_history = []
        
        # Clear document sources
        if "document_sources" in st.session_state:
            del st.session_state["document_sources"]
        
        st.success("Database cleared successfully!")
    
    st.divider()

# Main chat interface with integrated document selection and RAGate settings
st.header("💬 Chat")

# Create columns for chat controls
chat_controls_col1, chat_controls_col2 = st.columns([3, 1])

with chat_controls_col1:
    # Use loaded files list directly instead of querying the database
    if "loaded_files" in st.session_state:
        document_sources = st.session_state.loaded_files
    else:
        document_sources = []
    
    # Add "All Documents" option at the beginning
    options = ["All Documents"] + document_sources
    
    # Document selector integrated within chat interface
    selected_document = st.selectbox(
        "📑 Select document to query:",
        options=options,
        index=0,
        help="Choose 'All Documents' to search across all uploaded PDFs, or select a specific document."
    )

with chat_controls_col2:
    with st.expander("⚙️ RAGate", expanded=False):
        st.session_state.use_ragate = st.checkbox(
            "Use adaptive retrieval", 
            value=st.session_state.use_ragate,
            help="Enable/disable adaptive retrieval based on query type"
        )
        
        st.session_state.confidence_threshold = st.slider(
            "Confidence threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.confidence_threshold,
            step=0.05,
            help="Threshold for deciding when to use retrieval"
        )
        
        st.session_state.show_debug_info = st.checkbox(
            "Show debug info", 
            value=st.session_state.show_debug_info,
            help="Display RAGate decision information in responses"
        )

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new questions
if prompt := st.chat_input("Ask a question about your PDFs..."):
    # Add user message to chat history
    display_prompt = prompt
    
    # Add document context to the prompt if a specific document is selected
    if selected_document != "All Documents":
        display_prompt = f"[Regarding: {selected_document}] {prompt}"
    
    st.session_state.chat_history.append({"role": "user", "content": display_prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(display_prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        if not st.session_state.loaded_files:
            response = "Please upload PDF documents first before asking questions."
        else:
            try:
                # Initialize RAG chatbot with RAGate settings
                chatbot = RAGChatbot(
                    confidence_threshold=st.session_state.confidence_threshold,
                    use_ragate=st.session_state.use_ragate
                )
                
                # Get retrieval decision (for debug info)
                use_retrieval, confidence, explanation = chatbot.decide_retrieval(prompt)
                
                # Show debug info if enabled
                if st.session_state.show_debug_info:
                    st.info(f"RAGate: {explanation}")
                
                # Get relevant documents for the query
                with st.spinner("Searching for relevant information..."):
                    # If a specific document is selected, filter search by that document
                    source_filter = None if selected_document == "All Documents" else selected_document
                    relevant_docs = st.session_state.vector_store.similarity_search(
                        prompt, 
                        k=4, 
                        source_filter=source_filter
                    )
                
                # Show retrieved document context details in an expander
                with st.expander("View Retrieved Context", expanded=False):
                    st.markdown("### Retrieved Document Chunks")
                    for i, doc in enumerate(relevant_docs):
                        source = doc.metadata.get("source", "Unknown")
                        st.markdown(f"**Chunk {i+1}** from **{source}**")
                        st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                        st.divider()
                
                # Generate response with Gemini
                with st.spinner("Generating response..."):
                    response = chatbot.answer_question(prompt, relevant_docs)
            except Exception as e:
                response = f"Error: {str(e)}"
        
        response_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Show a welcome message for first-time users
if not st.session_state.chat_history:
    st.info("👋 Upload a PDF document and start asking questions!")
