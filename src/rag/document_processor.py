"""Document Processor for RAG"""

from src.rag.vector_store import VectorStore

class DocumentProcessor:
    """Processes documents and adds them to the vector store."""

    def __init__(self, vector_store: VectorStore):
        """Initializes the DocumentProcessor.

        Args:
            vector_store: A VectorStore object.
        """
        self.vector_store = vector_store

    def process_document(self, document_path: str):
        """Processes a document, extracts text, creates embeddings, and adds to the vector store.

        Args:
            document_path: The path to the document to process.
        """
        # In a real implementation, this would involve:
        # 1. Reading the document (PDF, text, etc.).
        # 2. Splitting the text into chunks.
        # 3. Generating embeddings for each chunk.
        # 4. Adding the embeddings to the vector store.
        print(f"Processing document at {document_path}...")

        # Dummy data for demonstration
        text_chunks = ["This is a chunk about crop rotation.", "This is a chunk about pest control."]
        for i, chunk in enumerate(text_chunks):
            # Dummy embedding
            embedding = [0.1, 0.2, 0.3]
            self.vector_store.add(f"doc_{i}", embedding, {"text": chunk})

        print(f"Document {document_path} processed and added to the vector store.")