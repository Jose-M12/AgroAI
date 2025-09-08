"""Vector Store for RAG"""

class VectorStore:
    """A simple in-memory vector store for demonstration purposes."""

    def __init__(self):
        """Initializes the VectorStore."""
        self.vectors = {}

    def add(self, vector_id: str, vector: list, metadata: dict):
        """Adds a vector to the store.

        Args:
            vector_id: The ID of the vector.
            vector: The vector to add.
            metadata: Metadata associated with the vector.
        """
        self.vectors[vector_id] = {"vector": vector, "metadata": metadata}

    def search(self, query_vector: list, top_k: int = 5) -> list:
        """Searches for the most similar vectors to a query vector.

        Args:
            query_vector: The query vector.
            top_k: The number of results to return.

        Returns:
            A list of the top_k most similar vectors.
        """
        # This is a placeholder for a real similarity search.
        # In a real implementation, you would use a library like Faiss or a managed service like Pinecone.
        print(f"Searching for similar vectors...")
        # Returning some dummy data for now
        return ["Document 1 about crop rotation", "Document 2 about pest control"]