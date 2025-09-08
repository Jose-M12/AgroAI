# AgroAI Colombia: Comprehensive Documentation

## 1. General Project Overview

### Purpose and Functionality

AgroAI Colombia is an intelligent agricultural advisory system designed to assist Colombian farmers with crop management, pest detection, and climate-adaptive farming strategies. The project leverages generative AI to provide real-time, data-driven recommendations to farmers, helping them to improve crop yields, reduce costs, and mitigate the effects of climate change.

The system is built around a set of intelligent agents, each specializing in a different area of agriculture:

*   **Crop Specialist Agent**: Recommends the best crops to plant based on location, soil type, and season.
*   **Pest Detection Agent**: Identifies pests and diseases from images of plants.
*   **Climate Advisor Agent**: Provides weather forecasts and climate adaptation strategies.
*   **Market Intelligence Agent**: Analyzes market trends and predicts crop prices.
*   **Knowledge Synthesis Agent**: Combines the information from all the other agents and the RAG system to provide comprehensive advice to the farmer.

The project also includes a Retrieval-Augmented Generation (RAG) system that provides the agents with access to a vast knowledge base of agricultural research papers, climate data, and best practices.

The entire system is exposed through a FastAPI application, which provides a set of API endpoints that can be used to interact with the agents and the RAG system.

## 2. Installation Instructions

To set up and run the project locally, follow these steps:

### Prerequisites

*   Anaconda or Miniconda
*   Python 3.9 or higher

### Using Anaconda/Miniconda (Recommended)

1.  **Create and activate the Conda environment:**

    ```bash
    conda env create -f environment.yml
    conda activate agroai
    ```

### Using pip

1.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### 3. Run the Application

Start the FastAPI application using `uvicorn`:

```bash
uvicorn src.api.fastapi_app:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## 3. Code Explanation

This section provides a detailed explanation of the project's source code.

### 3.1. `src/api/fastapi_app.py`

This file is the main entry point of the application. It creates the FastAPI application and defines the API endpoints.

```python
"""FastAPI application"""
from fastapi import FastAPI
from pydantic import BaseModel
from src.agents.crop_specialist import CropSpecialistAgent
from src.agents.pest_detection import PestDetectionAgent
from src.agents.climate_advisor import ClimateAdvisorAgent
from src.agents.market_intelligence import MarketIntelligenceAgent
from src.agents.knowledge_synthesis import KnowledgeSynthesisAgent
from src.rag.retrieval_engine import RetrievalEngine
from src.rag.vector_store import VectorStore
from src.rag.document_processor import DocumentProcessor
from src.rag.context_manager import ContextManager


app = FastAPI()

# Initialize agents and other components at startup
vector_store = VectorStore()
document_processor = DocumentProcessor(vector_store)
retrieval_engine = RetrievalEngine(vector_store)
context_manager = ContextManager()
crop_specialist = CropSpecialistAgent()
pest_detector = PestDetectionAgent()
climate_advisor = ClimateAdvisorAgent()
market_intel = MarketIntelligenceAgent()
knowledge_synthesizer = KnowledgeSynthesisAgent(retrieval_engine)

class CropRecommendationRequest(BaseModel):
    location: str
    soil_type: str
    season: str

@app.get("/")
def read_root():
    return {"message": "Welcome to AgroAI Colombia"}

@app.post("/recommend_crop")
def recommend_crop(request: CropRecommendationRequest):
    """Recommends a crop based on location, soil type, and season."""
    recommendation = crop_specialist.recommend_crop(request.location, request.soil_type, request.season)
    return {"recommendation": recommendation}

class PestDetectionRequest(BaseModel):
    image_path: str

@app.post("/detect_pest")
def detect_pest(request: PestDetectionRequest):
    """Detects pests in an image."""
    detection = pest_detector.detect_pest(request.image_path)
    return {"detection": detection}

class WeatherForecastRequest(BaseModel):
    location: str

@app.post("/weather_forecast")
def get_weather_forecast(request: WeatherForecastRequest):
    """Gets the weather forecast for a specific location."""
    forecast = climate_advisor.get_weather_forecast(request.location)
    return {"forecast": forecast}

class MarketPriceRequest(BaseModel):
    crop: str

@app.post("/market_price")
def get_market_price(request: MarketPriceRequest):
    """Gets the market price for a crop."""
    price = market_intel.get_market_price(request.crop)
    return {"price": price}

class AdviceRequest(BaseModel):
    user_id: str
    image_path: str

@app.post("/generate_advice")
def generate_advice(request: AdviceRequest):
    """Generates comprehensive advice for a user."""
    # Get user context
    user_context = context_manager.get_context(request.user_id)

    # Run the agents
    crop_recommendation = crop_specialist.recommend_crop(user_context['location'], user_context['soil_type'], user_context['season'])
    pest_detection = pest_detector.detect_pest(request.image_path)
    weather_forecast = climate_advisor.get_weather_forecast(user_context['location'])
    market_price = market_intel.get_market_price(crop_recommendation)

    # Generate advice
    advice = knowledge_synthesizer.generate_advice(crop_recommendation, pest_detection, weather_forecast, market_price)

    return {"advice": advice}
```

**Line-by-Line Explanation:**

*   **Lines 1-12**: These lines import the necessary classes from the `fastapi`, `pydantic`, and other modules in the project.
*   **Line 15**: This line creates an instance of the `FastAPI` class, which is the main object for the application.
*   **Lines 18-27**: These lines initialize all the agents and RAG components at startup. This is an efficient way to manage these components, as they are created only once when the application starts, and then reused for each request.
*   **Lines 29-33**: This defines a Pydantic model for the request body of the `/recommend_crop` endpoint. It specifies that the request body must be a JSON object with `location`, `soil_type`, and `season` fields.
*   **Lines 35-38**: This defines a GET endpoint for the root URL (`/`). It returns a simple welcome message.
*   **Lines 40-44**: This defines a POST endpoint for `/recommend_crop`. It takes a `CropRecommendationRequest` object as input, calls the `recommend_crop` method of the `CropSpecialistAgent`, and returns the recommendation.
*   **Lines 46-49**: This defines a Pydantic model for the request body of the `/detect_pest` endpoint.
*   **Lines 51-55**: This defines a POST endpoint for `/detect_pest`. It takes a `PestDetectionRequest` object as input, calls the `detect_pest` method of the `PestDetectionAgent`, and returns the detection results.
*   **Lines 57-60**: This defines a Pydantic model for the request body of the `/weather_forecast` endpoint.
*   **Lines 62-66**: This defines a POST endpoint for `/weather_forecast`. It takes a `WeatherForecastRequest` object as input, calls the `get_weather_forecast` method of the `ClimateAdvisorAgent`, and returns the forecast.
*   **Lines 68-71**: This defines a Pydantic model for the request body of the `/market_price` endpoint.
*   **Lines 73-77**: This defines a POST endpoint for `/market_price`. It takes a `MarketPriceRequest` object as input, calls the `get_market_price` method of the `MarketIntelligenceAgent`, and returns the price.
*   **Lines 79-83**: This defines a Pydantic model for the request body of the `/generate_advice` endpoint.
*   **Lines 85-99**: This defines a POST endpoint for `/generate_advice`. This is the most complex endpoint, as it orchestrates all the other agents to generate comprehensive advice for the user. It gets the user context, runs the different agents, and then uses the `KnowledgeSynthesisAgent` to generate the final advice.

### 3.2. `src/agents/crop_specialist.py`

This file defines the `CropSpecialistAgent`, which is responsible for recommending crops and analyzing their growth stage.

```python
"""Crop Specialist Agent"""

class CropSpecialistAgent:
    """Agent specializing in crop recommendations and growth stage analysis."""

    def __init__(self, knowledge_base=None):
        """Initializes the CropSpecialistAgent.

        Args:
            knowledge_base: A knowledge base object to query for crop information.
        """
        self.knowledge_base = knowledge_base

    def recommend_crop(self, location: str, soil_type: str, season: str) -> str:
        """Recommends a crop based on location, soil type, and season.

        Args:
            location: The geographical location.
            soil_type: The type of soil.
            season: The current season.

        Returns:
            A string containing the recommended crop.
        """
        # In a real implementation, this would query the knowledge base
        # and use a more sophisticated model for recommendations.
        if soil_type == "loamy" and season == "rainy":
            return "Maize"
        elif soil_type == "clay" and season == "dry":
            return "Sorghum"
        else:
            return "Potato"

    def analyze_growth_stage(self, image_path: str) -> str:
        """Analyzes the growth stage of a crop from an image.

        Args:
            image_path: The path to the image of the crop.

        Returns:
            A string indicating the growth stage.
        """
        # This would typically involve a computer vision model.
        print(f"Analyzing image at {image_path}...")
        return "Vegetative Stage"
```

**Line-by-Line Explanation:**

*   **Line 3**: Defines the `CropSpecialistAgent` class.
*   **Lines 6-12**: The constructor (`__init__`) initializes the agent. It takes an optional `knowledge_base` argument, which would be used in a real implementation to query for crop information.
*   **Lines 14-29**: The `recommend_crop` method takes a location, soil type, and season as input and returns a crop recommendation. The current implementation uses a simple set of rules, but in a real application, this would involve a more sophisticated model.
*   **Lines 31-41**: The `analyze_growth_stage` method takes an image path as input and returns the growth stage of the crop. This is a placeholder for a computer vision model.

### 3.3. `src/agents/pest_detection.py`

This file defines the `PestDetectionAgent`, which is responsible for detecting pests and identifying diseases from images.

```python
"""Pest Detection Agent"""

class PestDetectionAgent:
    """Agent specializing in pest detection and disease identification."""

    def __init__(self, model_path: str = None):
        """Initializes the PestDetectionAgent.

        Args:
            model_path: The path to the trained computer vision model.
        """
        self.model_path = model_path
        # In a real implementation, you would load the model here.
        # self.model = self.load_model(model_path)

    def detect_pest(self, image_path: str) -> dict:
        """Detects pests in an image.

        Args:
            image_path: The path to the image to analyze.

        Returns:
            A dictionary containing the detected pests and their confidence scores.
        """
        # This would use a computer vision model (e.g., YOLOv8) for detection.
        print(f"Analyzing image at {image_path} for pests...")
        return {"pest": "aphid", "confidence": 0.95}

    def identify_disease(self, image_path: str) -> dict:
        """Identifies diseases in an image of a plant.

        Args:
            image_path: The path to the image to analyze.

        Returns:
            A dictionary containing the identified disease and confidence score.
        """
        print(f"Analyzing image at {image_path} for diseases...")
        return {"disease": "powdery_mildew", "confidence": 0.88}
```

**Line-by-Line Explanation:**

*   **Line 3**: Defines the `PestDetectionAgent` class.
*   **Lines 6-12**: The constructor (`__init__`) initializes the agent. It takes an optional `model_path` argument, which would be the path to a trained computer vision model.
*   **Lines 14-24**: The `detect_pest` method takes an image path as input and returns a dictionary with the detected pest and a confidence score. This is a placeholder for a real computer vision model.
*   **Lines 26-36**: The `identify_disease` method takes an image path as input and returns a dictionary with the identified disease and a confidence score. This is also a placeholder for a real computer vision model.

### 3.4. `src/agents/climate_advisor.py`

This file defines the `ClimateAdvisorAgent`, which is responsible for providing weather forecasts and climate adaptation strategies.

```python
"""Climate Advisor Agent"""

class ClimateAdvisorAgent:
    """Agent specializing in weather pattern analysis and climate adaptation strategies."""

    def __init__(self, weather_api_key: str = None):
        """Initializes the ClimateAdvisorAgent.

        Args:
            weather_api_key: The API key for a weather service.
        """
        self.weather_api_key = weather_api_key

    def get_weather_forecast(self, location: str) -> dict:
        """Gets the weather forecast for a specific location.

        Args:
            location: The location for which to get the forecast.

        Returns:
            A dictionary containing the weather forecast.
        """
        # In a real implementation, this would call a weather API (e.g., IDEAM).
        print(f"Getting weather forecast for {location}...")
        return {
            "temperature": "25°C",
            "humidity": "70%",
            "precipitation": "10%"
        }

    def recommend_adaptation_strategy(self, weather_data: dict) -> str:
        """Recommends a climate adaptation strategy based on weather data.

        Args:
            weather_data: A dictionary containing weather data.

        Returns:
            A string with the recommended adaptation strategy.
        """
        if int(weather_data["temperature"].replace("°C", "")) > 30:
            return "Consider using shade cloths to protect crops from excessive heat."
        else:
            return "Current weather conditions are optimal."
```

**Line-by-Line Explanation:**

*   **Line 3**: Defines the `ClimateAdvisorAgent` class.
*   **Lines 6-12**: The constructor (`__init__`) initializes the agent. It takes an optional `weather_api_key` argument, which would be used to authenticate with a weather API.
*   **Lines 14-26**: The `get_weather_forecast` method takes a location as input and returns a dictionary with the weather forecast. This is a placeholder for a real weather API call.
*   **Lines 28-38**: The `recommend_adaptation_strategy` method takes a dictionary of weather data as input and returns a climate adaptation strategy. The current implementation uses a simple rule, but a real application would likely use more complex logic.

### 3.5. `src/agents/market_intelligence.py`

This file defines the `MarketIntelligenceAgent`, which is responsible for analyzing market trends and predicting crop prices.

```python
"""Market Intelligence Agent"""

class MarketIntelligenceAgent:
    """Agent specializing in market trend analysis and price prediction."""

    def __init__(self, market_data_api_key: str = None):
        """Initializes the MarketIntelligenceAgent.

        Args:
            market_data_api_key: The API key for a market data service.
        """
        self.market_data_api_key = market_data_api_key

    def get_market_price(self, crop: str) -> dict:
        """Gets the current market price for a crop.

        Args:
            crop: The crop to get the price for.

        Returns:
            A dictionary containing the crop's market price.
        """
        # In a real implementation, this would call a market data API.
        print(f"Getting market price for {crop}...")
        prices = {
            "Maize": {"price": "$500/ton", "trend": "up"},
            "Sorghum": {"price": "$450/ton", "trend": "stable"},
            "Potato": {"price": "$600/ton", "trend": "down"}
        }
        return prices.get(crop, {"price": "N/A", "trend": "N/A"})

    def predict_market_trend(self, crop: str) -> str:
        """Predicts the market trend for a crop.

        Args:
            crop: The crop to predict the trend for.

        Returns:
            A string indicating the predicted market trend.
        """
        # This would use a predictive model based on historical data.
        print(f"Predicting market trend for {crop}...")
        trends = {
            "Maize": "The price is expected to increase in the next quarter.",
            "Sorghum": "The price will likely remain stable.",
            "Potato": "A decrease in price is expected due to high supply."
        }
        return trends.get(crop, "No prediction available.")
```

**Line-by-Line Explanation:**

*   **Line 3**: Defines the `MarketIntelligenceAgent` class.
*   **Lines 6-12**: The constructor (`__init__`) initializes the agent. It takes an optional `market_data_api_key` argument, which would be used to authenticate with a market data API.
*   **Lines 14-26**: The `get_market_price` method takes a crop name as input and returns a dictionary with the market price and trend. This is a placeholder for a real market data API call.
*   **Lines 28-38**: The `predict_market_trend` method takes a crop name as input and returns a string with the predicted market trend. This is a placeholder for a real predictive model.

### 3.6. `src/agents/knowledge_synthesis.py`

This file defines the `KnowledgeSynthesisAgent`, which is the final agent in the chain. It is responsible for taking the outputs of all the other agents and synthesizing them into a single, comprehensive piece of advice for the user.

```python
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
```

**Line-by-Line Explanation:**

*   **Line 3**: Imports the `RetrievalEngine` class from the RAG system.
*   **Line 5**: Defines the `KnowledgeSynthesisAgent` class.
*   **Lines 8-14**: The constructor (`__init__`) initializes the agent. It takes a `RetrievalEngine` object as an argument, which it will use to retrieve additional information from the knowledge base.
*   **Lines 16-34**: The `generate_advice` method takes the outputs of the other agents as input. It uses the `RetrievalEngine` to get additional context, and then constructs a comprehensive piece of advice for the user. In a real application, this would likely involve a large language model (LLM) to generate more natural and human-like advice.

### 3.7. `src/rag/context_manager.py`

This file defines the `ContextManager`, which is responsible for managing the context for the RAG system.

```python
"""Context Manager for RAG"""

class ContextManager:
    """Manages the context for the RAG system, including geo-spatial, temporal, and agricultural context."""

    def __init__(self):
        """Initializes the ContextManager."""
        pass

    def get_context(self, user_id: str) -> dict:
        """Gets the context for a specific user.

        Args:
            user_id: The ID of the user.

        Returns:
            A dictionary containing the user's context.
        """
        # In a real implementation, this would fetch context from various sources
        # like user profiles, device location, and external APIs.
        print(f"Getting context for user {user_id}...")
        return {
            "location": "Boyaca, Colombia",
            "soil_type": "loamy",
            "season": "rainy",
            "historical_yield": "5 tons/hectare"
        }
```

**Line-by-Line Explanation:**

*   **Line 3**: Defines the `ContextManager` class.
*   **Lines 6-9**: The constructor (`__init__`) initializes the context manager.
*   **Lines 11-23**: The `get_context` method takes a user ID as input and returns a dictionary with the user's context. This is a placeholder for a real implementation that would fetch context from various sources.

### 3.8. `src/rag/document_processor.py`

This file defines the `DocumentProcessor`, which is responsible for processing documents and adding them to the vector store.

```python
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
```

**Line-by-Line Explanation:**

*   **Line 3**: Imports the `VectorStore` class.
*   **Line 5**: Defines the `DocumentProcessor` class.
*   **Lines 8-14**: The constructor (`__init__`) initializes the document processor. It takes a `VectorStore` object as an argument, which it will use to store the processed documents.
*   **Lines 16-30**: The `process_document` method takes a document path as input, processes the document, and adds it to the vector store. The current implementation is a placeholder and shows the basic steps involved in processing a document.

### 3.9. `src/rag/vector_store.py`

This file defines the `VectorStore`, which is a simple in-memory vector store for demonstration purposes.

```python
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
```

**Line-by-Line Explanation:**

*   **Line 3**: Defines the `VectorStore` class.
*   **Lines 6-9**: The constructor (`__init__`) initializes the vector store as an empty dictionary.
*   **Lines 11-18**: The `add` method takes a vector ID, a vector, and metadata as input and adds them to the vector store.
*   **Lines 20-31**: The `search` method takes a query vector as input and returns a list of the most similar vectors. This is a placeholder for a real similarity search.

### 3.10. `src/rag/retrieval_engine.py`

This file defines the `RetrievalEngine`, which is responsible for retrieving information from the vector store.

```python
"""Retrieval Engine for RAG"""

from .vector_store import VectorStore

class RetrievalEngine:
    """Engine for retrieving information from the vector store."""

    def __init__(self, vector_store: VectorStore):
        """Initializes the RetrievalEngine.

        Args:
            vector_store: A VectorStore object.
        """
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """Retrieves the top_k most relevant documents for a query.

        Args:
            query: The query to search for.
            top_k: The number of documents to retrieve.

        Returns:
            A string containing the concatenated content of the retrieved documents.
        """
        # In a real implementation, this would query the vector store.
        print(f"Retrieving documents for query: '{query}'")
        # In a real implementation, you would generate an embedding for the query.
        query_vector = [0.1, 0.2, 0.3] # Dummy vector
        results = self.vector_store.search(query_vector, top_k)
        return "\n".join(results)
```

**Line-by-Line Explanation:**

*   **Line 3**: Imports the `VectorStore` class.
*   **Line 5**: Defines the `RetrievalEngine` class.
*   **Lines 8-14**: The constructor (`__init__`) initializes the retrieval engine. It takes a `VectorStore` object as an argument.
*   **Lines 16-28**: The `retrieve` method takes a query string as input, generates a dummy vector for the query, and then uses the `VectorStore` to find the most similar documents. It returns a string containing the concatenated content of the retrieved documents.