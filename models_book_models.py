"""
Book and Agent Models - Pydantic Models for API and Validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid

# Enums
class BookStatus(str, Enum):
    PLANNING = "planning"
    RESEARCHING = "researching" 
    WRITING = "writing"
    EDITING = "editing"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentType(str, Enum):
    CATEGORY_SELECTION = "category_selection"
    RESEARCH_PLANNING = "research_planning"
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    FACT_CHECKING = "fact_checking"
    WRITING = "writing"
    CONTENT_GENERATION = "content_generation"
    ILLUSTRATION = "illustration"
    EDITING_QA = "editing_qa"
    PUBLICATION = "publication"

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class BookFormat(str, Enum):
    PDF = "pdf"
    EPUB = "epub"
    HTML = "html"
    DOCX = "docx"

class WritingStyle(str, Enum):
    TECHNICAL = "technical"
    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"
    CREATIVE = "creative"
    INFORMATIVE = "informative"

# Request Models
class BookGenerationRequest(BaseModel):
    """Request model for book generation"""
    title: str = Field(..., min_length=3, max_length=500, description="Book title")
    topic: Optional[str] = Field(None, min_length=3, max_length=500, description="Book topic/subject (auto-generated from title if not provided)")
    category: str = Field(..., description="Book category")
    genre: Optional[str] = Field(None, description="Book genre (fiction, non-fiction, etc)")
    author: Optional[str] = Field(default="AI Generated", description="Author name")
    language: str = Field(default="en", description="Book language")
    target_audience: Optional[str] = Field(None, max_length=200, description="Target audience")
    writing_style: Optional[WritingStyle] = Field(default=WritingStyle.INFORMATIVE, description="Writing style")
    chapter_count: int = Field(default=8, ge=3, le=20, description="Number of chapters")
    target_length: int = Field(default=15000, ge=5000, le=50000, description="Target word count")
    
    # Advanced options
    user_preferences: Optional[Dict[str, Any]] = Field(default={}, description="User preferences")
    series_info: Optional[Dict[str, Any]] = Field(default=None, description="Series information")
    custom_requirements: Optional[str] = Field(None, description="Custom requirements")
    
    # Format options
    output_formats: List[BookFormat] = Field(default=[BookFormat.PDF], description="Output formats")
    include_illustrations: bool = Field(default=True, description="Include illustrations")
    
    @validator('category')
    def validate_category(cls, v):
        allowed_categories = ['technology', 'business', 'health', 'finance', 'education', 'lifestyle']
        if v not in allowed_categories:
            raise ValueError(f'Category must be one of: {allowed_categories}')
        return v
    
    @validator('topic', always=True)
    def auto_generate_topic(cls, v, values):
        """Auto-generate topic from title if not provided"""
        if v is None and 'title' in values:
            return values['title']
        return v

class ChapterOutlineRequest(BaseModel):
    """Request for chapter outline generation"""
    book_project_id: str
    chapter_number: int = Field(..., ge=1)
    chapter_title: str = Field(..., min_length=3, max_length=300)
    key_concepts: List[str] = Field(default=[])
    research_requirements: Optional[str] = Field(None)

class ResearchRequest(BaseModel):
    """Request for research data collection"""
    book_project_id: str
    chapter_id: Optional[str] = None
    research_topics: List[str] = Field(..., min_items=1)
    source_types: List[str] = Field(default=["web", "academic"])
    max_sources: int = Field(default=20, ge=1, le=100)
    language: str = Field(default="en")

# Response Models
class BookProjectResponse(BaseModel):
    """Response model for book project"""
    id: str
    title: str
    category: str
    language: str
    status: BookStatus
    progress_percentage: float
    current_stage: Optional[str]
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    
    # Configuration
    chapter_count: int
    target_length: int
    writing_style: Optional[str]
    target_audience: Optional[str]
    
    # Quality metrics
    quality_score: Optional[float]
    
    class Config:
        from_attributes = True

class ChapterResponse(BaseModel):
    """Response model for book chapter"""
    id: str
    book_project_id: str
    chapter_number: int
    title: str
    content: Optional[str]
    outline: Optional[str]
    status: str
    word_count: int
    quality_score: Optional[float]
    readability_score: Optional[float]
    
    # Metadata
    key_concepts: Optional[List[str]]
    fact_check_status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class AgentTaskResponse(BaseModel):
    """Response model for agent task"""
    id: str
    book_project_id: str
    agent_type: AgentType
    task_name: str
    status: TaskStatus
    
    # Performance
    execution_time: Optional[float]
    retry_count: int
    quality_score: Optional[float]
    
    # Data
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    # Error handling
    error_message: Optional[str]
    
    class Config:
        from_attributes = True

class ResearchDataResponse(BaseModel):
    """Response model for research data"""
    id: str
    book_project_id: str
    source_type: str
    source_url: Optional[str]
    source_title: Optional[str]
    
    content: str
    summary: Optional[str]
    key_points: Optional[List[str]]
    
    # Quality metrics
    credibility_score: Optional[float]
    relevance_score: Optional[float]
    fact_checked: bool
    verification_status: str
    
    # Metadata
    language: Optional[str]
    publish_date: Optional[datetime]
    author: Optional[str]
    collected_at: datetime
    
    class Config:
        from_attributes = True

class PublicationResponse(BaseModel):
    """Response model for book publication"""
    id: str
    book_project_id: str
    format_type: BookFormat
    file_path: str
    file_size: int
    
    # Publication metadata
    isbn: Optional[str]
    publisher: Optional[str]
    publication_date: Optional[datetime]
    version: str
    
    # Quality and distribution
    final_quality_score: Optional[float]
    validation_passed: bool
    is_public: bool
    download_count: int
    
    created_at: datetime
    
    class Config:
        from_attributes = True

# Agent Communication Models
class AgentMessage(BaseModel):
    """Message between agents"""
    sender_agent: str
    recipient_agent: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: int = Field(default=1, ge=1, le=5)

class AgentState(BaseModel):
    """Agent state information"""
    agent_id: str
    agent_type: AgentType
    status: TaskStatus
    current_task: Optional[str]
    
    # Performance metrics
    tasks_completed: int = 0
    average_execution_time: Optional[float]
    success_rate: Optional[float]
    
    # Resource usage
    memory_usage: Optional[float]
    cpu_usage: Optional[float]
    
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class WorkflowState(BaseModel):
    """Overall workflow state"""
    book_project_id: str
    current_stage: str
    progress_percentage: float
    
    # Active agents
    active_agents: List[str]
    pending_tasks: int
    completed_tasks: int
    failed_tasks: int
    
    # Quality metrics
    overall_quality_score: Optional[float]
    quality_issues: List[str] = []
    
    # Timing
    started_at: datetime
    estimated_completion: Optional[datetime]
    
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# Utility Models
class QualityMetrics(BaseModel):
    """Quality assessment metrics"""
    overall_score: float = Field(..., ge=0.0, le=1.0)
    
    # Content quality
    content_accuracy: Optional[float]
    readability: Optional[float]
    engagement: Optional[float]
    consistency: Optional[float]
    
    # Technical quality
    grammar_score: Optional[float]
    fact_check_score: Optional[float]
    plagiarism_score: Optional[float]
    
    # Issues
    critical_issues: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

class ProgressUpdate(BaseModel):
    """Progress update model"""
    book_project_id: str
    stage: str
    progress_percentage: float
    message: str
    
    # Additional data
    metrics: Optional[Dict[str, Any]]
    estimated_time_remaining: Optional[int]  # seconds
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorReport(BaseModel):
    """Error reporting model"""
    error_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    book_project_id: Optional[str]
    agent_type: Optional[AgentType]
    task_id: Optional[str]
    
    error_type: str
    error_message: str
    error_details: Optional[Dict[str, Any]]
    
    # Context
    stack_trace: Optional[str]
    input_data: Optional[Dict[str, Any]]
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: str = Field(default="error")  # info, warning, error, critical

# Configuration Models
class AgentConfig(BaseModel):
    """Agent configuration model"""
    agent_type: AgentType
    max_concurrent_tasks: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=1800, ge=60, le=7200)
    retry_attempts: int = Field(default=3, ge=0, le=5)
    
    # LLM settings
    llm_provider: str = Field(default="groq")
    model_name: str = Field(default="llama-3.1-8b-instant")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=100, le=8000)
    
    # Custom parameters
    custom_params: Dict[str, Any] = Field(default={})

class SystemConfig(BaseModel):
    """System-wide configuration"""
    max_concurrent_books: int = Field(default=5, ge=1, le=20)
    default_quality_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    
    # Resource limits
    max_chapters_per_book: int = Field(default=20, ge=3, le=50)
    max_word_count: int = Field(default=50000, ge=5000, le=100000)
    
    # Processing settings
    enable_parallel_processing: bool = Field(default=True)
    batch_size: int = Field(default=5, ge=1, le=20)
    
    # Storage settings
    max_file_size_mb: int = Field(default=100, ge=1, le=500)
    cleanup_after_days: int = Field(default=30, ge=1, le=365)

# Search and Filter Models
class BookSearchRequest(BaseModel):
    """Book search request"""
    query: Optional[str] = None
    category: Optional[str] = None
    status: Optional[BookStatus] = None
    language: Optional[str] = None
    
    # Filters
    min_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    
    # Pagination
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

class BookSearchResponse(BaseModel):
    """Book search response"""
    books: List[BookProjectResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int