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
