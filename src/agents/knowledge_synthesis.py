"""Knowledge Synthesis Agent"""

from src.rag.retrieval_engine import RetrievalEngine

class KnowledgeSynthesisAgent:
    """Agent responsible for RAG-based advice generation and multi-modal response creation."""

    def __init__(self, retrieval_engine: RetrievalEngine):
        """Initializes the KnowledgeSynthesisAgent.

        Args:
            retrieval_engine: A RetrievalEngine object for the RAG system.
        """
        self.retrieval_engine = retrieval_engine

    def generate_advice(self, crop_recommendation: str, pest_detection: dict, weather_forecast: dict, market_price: dict) -> str:
        """Generates comprehensive advice based on inputs from other agents.

        Args:
            crop_recommendation: The recommended crop.
            pest_detection: The detected pests.
            weather_forecast: The weather forecast.
            market_price: The market price of the crop.

        Returns:
            A string containing synthesized advice.
        """
        # Use the retrieval engine to get additional context.
        context = self.retrieval_engine.retrieve(f"{crop_recommendation} farming practices")

        # Generate a response using a large language model (LLM).
        # This is a simplified example.
        advice = f"Based on the current conditions, we recommend planting {crop_recommendation}.\n"
        advice += f"The weather forecast is {weather_forecast['temperature']} with {weather_forecast['humidity']} humidity.\n"
        advice += f"The current market price for {crop_recommendation} is {market_price['price']}.\n"
        if pest_detection['pest']:
            advice += f"Be aware of {pest_detection['pest']} pests in your area.\n"
        advice += f"\nAdditional information: {context}"

        return advice