"""
Main FastAPI Application - Agentic RAG Book Generator
"""
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json

from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

# Configuration imports
from config_settings import settings
from config_database import create_tables, database_health_check, get_db, database

# Model imports
from models_book_models import (
    BookGenerationRequest, BookProjectResponse, AgentTaskResponse,
    BookSearchRequest, BookSearchResponse, ProgressUpdate
)

# Service imports
from services_llm_service import LLMService
from services_embedding_service import EmbeddingService
from services_vector_db_service import VectorDBService
from services_web_scraping_service import WebScrapingService

# Agent imports
from agents_base_agent import AgentCoordinator
from agents_category_selection import CategorySelectionAgent
from agents_research_planning import ResearchPlanningAgent
from agents_knowledge_acquisition import KnowledgeAcquisitionAgent
from agents_fact_checking import FactCheckingAgent
from agents_content_generation import ContentGenerationAgent
from agents_editing_qa import EditingQAAgent
from agents_publication import PublicationAgent

# Orchestrator imports
from orchestrators_central import CentralOrchestrator

# Utility imports
import uvicorn
import builtins
builtins.open = open

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Global variables for services and agents
llm_service: Optional[LLMService] = None
embedding_service: Optional[EmbeddingService] = None
vector_db_service: Optional[VectorDBService] = None
web_scraping_service: Optional[WebScrapingService] = None
agent_coordinator: Optional[AgentCoordinator] = None
central_orchestrator: Optional[CentralOrchestrator] = None

# -------------------------
# JSON safety helper
# -------------------------
def json_safe(data):
    """Recursively convert datetimes and other non-serializable types to JSON-safe forms."""
    if isinstance(data, dict):
        return {k: json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [json_safe(v) for v in data]
    if isinstance(data, set):
        return [json_safe(v) for v in list(data)]
    if isinstance(data, datetime):
        return data.isoformat()
    # leave basic types as-is
    return data

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Agentic RAG Book Generator...")
    
    try:
        # Initialize database
        await create_tables()
        logger.info("Database initialized")
        
        # Initialize services
        await initialize_services()
        logger.info("Services initialized")
        
        # Initialize agents
        await initialize_agents()
        logger.info("Agents initialized")
        
        # Initialize orchestrator
        await initialize_orchestrator()
        logger.info("Orchestrator initialized")
        
        logger.info("🚀 Agentic RAG Book Generator started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agentic RAG Book Generator...")
    await shutdown_services()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Automated book generation system using multiple AI agents",
    version=settings.app_version,
    lifespan=lifespan
)

# Configure CORS - IMPORTANT: Must be before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic RAG Book Generator API",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    try:
        # Check database
        db_health = await database_health_check()
        
        # Check services
        services_health = {
            "llm_service": await llm_service.health_check() if llm_service else {"status": "not_initialized"},
            "embedding_service": await embedding_service.health_check() if embedding_service else {"status": "not_initialized"},
            "vector_db_service": await vector_db_service.health_check() if vector_db_service else {"status": "not_initialized"}
        }
        
        # Check agents
        agents_health = {}
        if agent_coordinator:
            agents_health = await agent_coordinator.get_system_status()
        
        overall_status = "healthy"
        # use .get to avoid KeyError when status key missing
        if (db_health.get("status") != "healthy" or 
            any(s.get("status") != "healthy" for s in services_health.values())):
            overall_status = "unhealthy"
        
        payload = {
            "status": overall_status,
            "timestamp": datetime.utcnow(),
            "database": db_health,
            "services": services_health,
            "agents": agents_health,
            "version": settings.app_version
        }
        
        return JSONResponse(content=json_safe(payload))
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        error_payload = {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=json_safe(error_payload)
        )

# Book Generation Endpoints
@app.post("/books/generate", response_model=Dict[str, Any])
async def generate_book(request: BookGenerationRequest, background_tasks: BackgroundTasks):
    """Start book generation process"""
    try:
        logger.info(f"Received book generation request: {request.title}")
        
        if not central_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )
        
        # Start book generation
        project_id = await central_orchestrator.start_book_generation(request)
        
        logger.info(f"Book generation started: {project_id}")
        
        return {
            "project_id": project_id,
            "status": "started",
            "message": "Book generation started successfully",
            "estimated_duration": "30-60 minutes",
            "webhook_url": f"/books/{project_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Failed to start book generation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start book generation: {str(e)}"
        )

@app.get("/books/{project_id}/status")
async def get_book_status(project_id: str):
    """Get book generation status"""
    try:
        if not central_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )
        
        status_info = await central_orchestrator.get_project_status(project_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        return JSONResponse(content=json_safe(status_info))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get project status: {str(e)}"
        )

@app.post("/books/{project_id}/pause")
async def pause_book_generation(project_id: str):
    """Pause book generation"""
    try:
        if not central_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )
        
        success = await central_orchestrator.pause_project(project_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or cannot be paused"
            )
        
        return {"message": "Project paused successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause project: {str(e)}"
        )

@app.post("/books/{project_id}/resume")
async def resume_book_generation(project_id: str):
    """Resume book generation"""
    try:
        if not central_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )
        
        success = await central_orchestrator.resume_project(project_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or cannot be resumed"
            )
        
        return {"message": "Project resumed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume project: {str(e)}"
        )

@app.delete("/books/{project_id}")
async def cancel_book_generation(project_id: str):
    """Cancel book generation"""
    try:
        if not central_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )
        
        success = await central_orchestrator.cancel_project(project_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        return {"message": "Project cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel project: {str(e)}"
        )

# ========================================
# UPDATED: Download & Export Endpoints
# ========================================

@app.get("/books/{project_id}/download/{format}")
async def download_book(project_id: str, format: str):
    """Download generated book in specified format - UPDATED to fetch from database"""
    try:
        # Validate format
        valid_formats = ["txt", "text", "md", "markdown", "json", "pdf", "epub", "html", "docx"]
        if format not in valid_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format. Supported formats: {valid_formats}"
            )
        
        # Get project info
        project_query = "SELECT * FROM projects WHERE id = :id"
        project = await database.fetch_one(project_query, {"id": project_id})
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Book not found"
            )
        
        # Check if book is completed
        if project['progress_percentage'] < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Book generation not completed yet"
            )
        
        # Get chapters
        chapters_query = """
            SELECT chapter_number, title, content, status, word_count
            FROM chapters
            WHERE project_id = :id
            ORDER BY chapter_number
        """
        chapters = await database.fetch_all(chapters_query, {"id": project_id})
        
        if not chapters:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No chapters found for this book"
            )
        
        # Check if we have a file-based format first (for legacy support)
        if format in ["pdf", "epub", "html", "docx"]:
            file_path = os.path.join(settings.output_path, project_id, f"book.{format}")
            if os.path.exists(file_path):
                return FileResponse(
                    file_path,
                    media_type=f"application/{format}",
                    filename=f"book_{project_id}.{format}"
                )
        
        # Route to appropriate export based on format (database-based)
        if format in ["txt", "text"]:
            return await _export_as_text(project, chapters)
        elif format in ["md", "markdown"]:
            return await _export_as_markdown(project, chapters)
        elif format == "json":
            return await _export_as_json(project, chapters)
        else:
            # For pdf, epub, html, docx - if no file exists, inform user
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Format '{format}' is not yet available. Try 'txt', 'md', or 'json' formats."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download book: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download book: {str(e)}"
        )


# ========================================
# NEW: Helper Functions for Exports
# ========================================

async def _export_as_text(project, chapters):
    """Export book as plain text"""
    text_parts = []
    text_parts.append("=" * 80)
    text_parts.append(f"\n{project['title']}\n")
    text_parts.append("=" * 80)
    text_parts.append(f"\n\nTopic: {project['topic']}")
    text_parts.append(f"\nAuthor: {project['author']}")
    text_parts.append(f"\nGenre: {project['genre']}")
    text_parts.append(f"\nCreated: {project['created_at']}\n")
    text_parts.append("\n" + "=" * 80 + "\n\n")
    
    for chapter in chapters:
        if chapter['content']:
            text_parts.append(f"\n\n{chapter['content']}\n\n")
            text_parts.append("=" * 80 + "\n\n")
        else:
            text_parts.append(f"\n\nChapter {chapter['chapter_number']}: {chapter['title']}\n")
            text_parts.append("[Content not yet generated]\n\n")
    
    text_content = "".join(text_parts)
    
    return Response(
        content=text_content.encode('utf-8'),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={project['title'].replace(' ', '_')}.txt"
        }
    )


async def _export_as_markdown(project, chapters):
    """Export book as markdown"""
    md_parts = []
    md_parts.append(f"# {project['title']}\n\n")
    md_parts.append(f"**Topic:** {project['topic']}  \n")
    md_parts.append(f"**Author:** {project['author']}  \n")
    md_parts.append(f"**Genre:** {project['genre']}  \n")
    md_parts.append(f"**Created:** {project['created_at']}  \n\n")
    md_parts.append("---\n\n")
    
    for chapter in chapters:
        if chapter['content']:
            md_parts.append(f"{chapter['content']}\n\n")
            md_parts.append("---\n\n")
        else:
            md_parts.append(f"## Chapter {chapter['chapter_number']}: {chapter['title']}\n\n")
            md_parts.append("*Content not yet generated*\n\n")
            md_parts.append("---\n\n")
    
    md_content = "".join(md_parts)
    
    return Response(
        content=md_content.encode('utf-8'),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={project['title'].replace(' ', '_')}.md"
        }
    )


async def _export_as_json(project, chapters):
    """Export book as JSON"""
    result = {
        "project": {
            "id": project['id'],
            "title": project['title'],
            "topic": project['topic'],
            "genre": project['genre'],
            "author": project['author'],
            "created_at": str(project['created_at']),
            "status": project['status'],
            "progress_percentage": project['progress_percentage']
        },
        "chapters": [
            {
                "chapter_number": ch['chapter_number'],
                "title": ch['title'],
                "content": ch['content'],
                "status": ch['status'],
                "word_count": ch['word_count']
            } 
            for ch in chapters
        ],
        "total_chapters": len(chapters),
        "total_words": sum(ch['word_count'] or 0 for ch in chapters)
    }
    
    return JSONResponse(
        content=json_safe(result),
        headers={
            "Content-Disposition": f"attachment; filename={project['title'].replace(' ', '_')}.json"
        }
    )


# ========================================
# EXISTING: Export Endpoints (kept for backward compatibility)
# ========================================

@app.get("/books/{project_id}/export/text")
async def export_book_as_text(project_id: str):
    """Export book as plain text from database (legacy endpoint)"""
    try:
        # Get project info
        project_query = "SELECT * FROM projects WHERE id = :id"
        project = await database.fetch_one(project_query, {"id": project_id})
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Book not found"
            )
        
        # Check if completed
        if project['progress_percentage'] < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Book generation not completed yet"
            )
        
        # Get chapters
        chapters_query = """
            SELECT chapter_number, title, content
            FROM chapters
            WHERE project_id = :id
            ORDER BY chapter_number
        """
        chapters = await database.fetch_all(chapters_query, {"id": project_id})
        
        if not chapters:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No chapters found for this book"
            )
        
        # Use the helper function
        return await _export_as_text(project, chapters)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export book as text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export book: {str(e)}"
        )


@app.get("/books/{project_id}/export/markdown")
async def export_book_as_markdown(project_id: str):
    """Export book as markdown from database (legacy endpoint)"""
    try:
        # Get project info
        project_query = "SELECT * FROM projects WHERE id = :id"
        project = await database.fetch_one(project_query, {"id": project_id})
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Book not found"
            )
        
        # Get chapters
        chapters_query = """
            SELECT chapter_number, title, content
            FROM chapters
            WHERE project_id = :id
            ORDER BY chapter_number
        """
        chapters = await database.fetch_all(chapters_query, {"id": project_id})
        
        if not chapters:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No chapters found for this book"
            )
        
        # Use the helper function
        return await _export_as_markdown(project, chapters)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export book as markdown: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export book: {str(e)}"
        )


@app.get("/books/{project_id}/view")
async def view_book_content(project_id: str):
    """View complete book with all chapters (JSON)"""
    try:
        # Get project info
        project_query = "SELECT * FROM projects WHERE id = :id"
        project = await database.fetch_one(project_query, {"id": project_id})
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Book not found"
            )
        
        # Get chapters
        chapters_query = """
            SELECT chapter_number, title, content, status, word_count
            FROM chapters
            WHERE project_id = :id
            ORDER BY chapter_number
        """
        chapters = await database.fetch_all(chapters_query, {"id": project_id})
        
        result = {
            "project": dict(project),
            "chapters": [dict(chapter) for chapter in chapters],
            "total_chapters": len(chapters),
            "total_words": sum(ch['word_count'] or 0 for ch in chapters)
        }
        
        return JSONResponse(content=json_safe(result))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to view book: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to view book: {str(e)}"
        )


@app.get("/books/list")
async def list_all_books():
    """List all books"""
    try:
        query = """
            SELECT id, title, topic, genre, author, status, 
                   progress_percentage, created_at
            FROM projects
            ORDER BY created_at DESC
        """
        books = await database.fetch_all(query)
        
        return JSONResponse(content=json_safe({
            "books": [dict(book) for book in books],
            "total": len(books)
        }))
        
    except Exception as e:
        logger.error(f"Failed to list books: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list books: {str(e)}"
        )

# Agent Management Endpoints
@app.get("/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    try:
        if not agent_coordinator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent coordinator not initialized"
            )
        
        status = await agent_coordinator.get_system_status()
        return JSONResponse(content=json_safe(status))
        
    except Exception as e:
        logger.error(f"Failed to get agents status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agents status: {str(e)}"
        )

@app.get("/agents/{agent_type}/health")
async def get_agent_health(agent_type: str):
    """Get health status of specific agent"""
    try:
        if not agent_coordinator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent coordinator not initialized"
            )
        
        agent = agent_coordinator.get_agent(agent_type)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_type}"
            )
        
        health = await agent.health_check()
        return JSONResponse(content=json_safe(health))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent health: {str(e)}"
        )

# System Management
@app.get("/system/metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        payload = {
            "active_projects": len(central_orchestrator.active_projects) if central_orchestrator else 0,
            "total_agents": len(agent_coordinator.agents) if agent_coordinator else 0,
            "uptime": "Implement uptime tracking",
            "memory_usage": "Implement memory tracking",
            "timestamp": datetime.utcnow()
        }
        return JSONResponse(content=json_safe(payload))
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics: {str(e)}"
        )

# Initialization Functions
async def initialize_services():
    """Initialize all services"""
    global llm_service, embedding_service, vector_db_service, web_scraping_service
    
    # Initialize LLM service
    llm_service = LLMService({
        "groq_api_key": settings.groq_api_key,
        "groq_model": settings.groq_model,
        "openai_api_key": settings.openai_api_key,
        "local_llm_enabled": settings.local_llm_enabled,
        "ollama_base_url": settings.ollama_base_url
    })
    
    # Initialize embedding service
    embedding_service = EmbeddingService({
        "provider": settings.embedding_provider,
        "model_name": settings.sentence_transformers_model
    })
    
    # Initialize vector database service
    vector_db_service = VectorDBService({
        "provider": settings.vector_db_provider,
        "faiss_index_path": settings.faiss_index_path,
        "dimension": settings.vector_dimension,
        "pinecone_api_key": settings.pinecone_api_key
    })
    
    # Initialize web scraping service
    web_scraping_service = WebScrapingService({
        "max_concurrent_requests": settings.max_concurrent_requests,
        "request_delay": settings.request_delay,
        "user_agent": settings.user_agent
    })

async def initialize_agents():
    """Initialize all agents"""
    global agent_coordinator
    
    agent_coordinator = AgentCoordinator()
    
    # Initialize and register all agents
    agents = [
        CategorySelectionAgent(llm_service),
        ResearchPlanningAgent(llm_service, web_scraping_service),
        KnowledgeAcquisitionAgent(llm_service, web_scraping_service, vector_db_service),
        FactCheckingAgent(llm_service, web_scraping_service),
        ContentGenerationAgent(llm_service, embedding_service),
        EditingQAAgent(llm_service),
        PublicationAgent(llm_service)
    ]
    
    for agent in agents:
        agent_coordinator.register_agent(agent)

async def initialize_orchestrator():
    """Initialize central orchestrator"""
    global central_orchestrator
    
    central_orchestrator = CentralOrchestrator(
        agent_coordinator,
        config={
            "max_concurrent_projects": settings.max_concurrent_agents,
            "quality_threshold": settings.min_quality_score
        }
    )

async def shutdown_services():
    """Shutdown all services"""
    logger.info("Shutting down services...")
    
    # Implement service cleanup if needed
    if vector_db_service:
        await vector_db_service.close()

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=settings.workers
    )
