"""
Application Settings Configuration - Agentic RAG Book Generator
"""
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
import os

class Settings(BaseSettings):
    # Application
    app_name: str = "Agentic RAG Book Generator"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = ["*"]

    # API Keys
    groq_api_key: str
    openai_api_key: Optional[str] = None
    stability_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    
    # LLM Configuration
    groq_model: str = "llama-3.1-8b-instant"
    default_llm_provider: str = "groq"
    local_llm_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Database Configuration (SQLite3)
    database_url: str = "sqlite:///./data/agentic_rag.db"
    async_database_url: Optional[str] = None
    
    # Vector Database Configuration
    vector_db_provider: str = "faiss"
    faiss_index_path: str = "./data/faiss_index"
    vector_dimension: int = 384
    
    # Embeddings Configuration
    embedding_provider: str = "sentence_transformers"
    sentence_transformers_model: str = "all-MiniLM-L6-v2"
    
    # Celery Configuration
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    task_timeout: int = 3600  # 1 hour
    soft_task_timeout: int = 3300  # 55 minutes
    max_retries: int = 3
    
    # Book Generation Settings
    default_chapter_length: int = 2500  # words
    max_chapters: int = 20
    supported_formats: List[str] = ["pdf", "epub", "html", "docx"]
    supported_languages: List[str] = ["en", "es", "fr", "de", "it", "pt"]
    
    # Quality Control
    min_quality_score: float = 0.8
    fact_check_threshold: float = 0.9
    plagiarism_threshold: float = 0.3
    
    # File Storage
    upload_path: str = "./uploads"
    output_path: str = "./outputs"
    temp_path: str = "./temp"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Web Scraping
    max_concurrent_requests: int = 10
    request_delay: float = 1.0
    user_agent: str = "AgenticRAG-BookGenerator/1.0"
    
    # Image Generation
    default_image_style: str = "digital_art"
    image_resolution: str = "1024x1024"
    max_images_per_chapter: int = 3
    
    # Publishing
    default_publisher: str = "Agentic RAG Publishers"
    default_isbn_prefix: str = "979-8"
    
    # Performance
    max_concurrent_agents: int = 5
    agent_timeout: int = 1800  # 30 minutes
    cache_ttl: int = 3600  # 1 hour
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # CrewAI Configuration
    crew_memory: bool = True
    crew_verbose: bool = True
    crew_embedder_provider: str = "sentence_transformers"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Book Categories Configuration
BOOK_CATEGORIES = {
    "technology": {
        "name": "Technology & Programming",
        "description": "Technical guides, programming tutorials, software development",
        "target_length": 150,  # pages
        "chapter_count": 8,
        "style": "technical"
    },
    "business": {
        "name": "Business & Entrepreneurship", 
        "description": "Business strategies, startup guides, management",
        "target_length": 120,
        "chapter_count": 7,
        "style": "professional"
    },
    "health": {
        "name": "Health & Wellness",
        "description": "Health tips, fitness guides, mental wellness",
        "target_length": 100,
        "chapter_count": 6,
        "style": "informative"
    },
    "finance": {
        "name": "Personal Finance",
        "description": "Money management, investing, financial planning",
        "target_length": 110,
        "chapter_count": 7,
        "style": "advisory"
    },
    "education": {
        "name": "Education & Learning",
        "description": "Learning strategies, skill development, academic guides",
        "target_length": 130,
        "chapter_count": 8,
        "style": "educational"
    },
    "lifestyle": {
        "name": "Lifestyle & Self-Help",
        "description": "Personal development, productivity, life skills",
        "target_length": 90,
        "chapter_count": 6,
        "style": "conversational"
    }
}

# Agent Configuration
AGENT_CONFIGS = {
    "category_selection": {
        "role": "Category Selection Specialist",
        "goal": "Select the most appropriate book category based on user preferences",
        "backstory": "Expert in content categorization and user preference analysis",
        "max_execution_time": 300
    },
    "research_planning": {
        "role": "Research & Planning Strategist", 
        "goal": "Create comprehensive book outlines and research plans",
        "backstory": "Experienced book planner with expertise in content structure",
        "max_execution_time": 600
    },
    "knowledge_acquisition": {
        "role": "Knowledge Research Specialist",
        "goal": "Gather and curate relevant information from multiple sources",
        "backstory": "Expert researcher with access to diverse information sources",
        "max_execution_time": 1200
    },
    "fact_checking": {
        "role": "Fact Verification Expert",
        "goal": "Verify accuracy and credibility of all information",
        "backstory": "Experienced fact-checker with rigorous verification methods",
        "max_execution_time": 900
    },
    "content_generation": {
        "role": "Content Creation Writer",
        "goal": "Generate engaging and well-structured book content",
        "backstory": "Professional writer with expertise in educational content",
        "max_execution_time": 1800
    },
    "editing_qa": {
        "role": "Quality Assurance Editor",
        "goal": "Review and refine content for quality and consistency",
        "backstory": "Senior editor with high standards for content quality",
        "max_execution_time": 900
    },
    "publication": {
        "role": "Publication Specialist",
        "goal": "Format and compile content into professional publications",
        "backstory": "Expert in multi-format publishing and layout design",
        "max_execution_time": 600
    }
}