"""
Database Configuration and Connection Management
"""
import logging
from typing import Optional
from databases import Database
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, Text, DateTime, Boolean
from sqlalchemy.sql import func
import os

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/agentic_rag.db")

# Create database instance
database = Database(DATABASE_URL)

# Create metadata
metadata = MetaData()

# Define tables
projects_table = Table(
    "projects",
    metadata,
    Column("id", String, primary_key=True),
    Column("title", String, nullable=False),
    Column("topic", String, nullable=True),
    Column("genre", String, nullable=True),
    Column("author", String, nullable=True),
    Column("category", String, nullable=True),
    Column("language", String, default="en"),
    Column("status", String, nullable=False),
    Column("progress_percentage", Integer, default=0),
    Column("current_stage", String, nullable=True),
    Column("chapter_count", Integer, default=8),
    Column("target_length", Integer, default=15000),
    Column("writing_style", String, nullable=True),
    Column("target_audience", String, nullable=True),
    Column("quality_score", Float, nullable=True),
    Column("created_at", String, nullable=False),
    Column("last_update", String, nullable=False),
    Column("completed_at", String, nullable=True),
    Column("error_message", Text, nullable=True),
)

agent_tasks_table = Table(
    "agent_tasks",
    metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False),
    Column("agent_type", String, nullable=False),
    Column("task_name", String, nullable=True),
    Column("status", String, nullable=False, default="completed"),  # Added default
    Column("result", Text, nullable=True),
    Column("execution_time", Float, nullable=True),
    Column("retry_count", Integer, default=0),
    Column("quality_score", Float, nullable=True),
    Column("created_at", String, nullable=False),
    Column("started_at", String, nullable=True),
    Column("completed_at", String, nullable=True),
    Column("error_message", Text, nullable=True),
)

chapters_table = Table(
    "chapters",
    metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False),
    Column("chapter_number", Integer, nullable=False),
    Column("title", String, nullable=False),
    Column("content", Text, nullable=True),
    Column("outline", Text, nullable=True),
    Column("status", String, nullable=False, default="pending"),
    Column("word_count", Integer, default=0),
    Column("quality_score", Float, nullable=True),
    Column("readability_score", Float, nullable=True),
    Column("fact_check_status", String, default="pending"),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)

research_data_table = Table(
    "research_data",
    metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False),
    Column("source_type", String, nullable=False),
    Column("source_url", String, nullable=True),
    Column("source_title", String, nullable=True),
    Column("content", Text, nullable=False),
    Column("summary", Text, nullable=True),
    Column("credibility_score", Float, nullable=True),
    Column("relevance_score", Float, nullable=True),
    Column("fact_checked", Boolean, default=False),
    Column("verification_status", String, default="pending"),
    Column("language", String, nullable=True),
    Column("author", String, nullable=True),
    Column("collected_at", String, nullable=False),
)

publications_table = Table(
    "publications",
    metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False),
    Column("format_type", String, nullable=False),
    Column("file_path", String, nullable=False),
    Column("file_size", Integer, nullable=False),
    Column("isbn", String, nullable=True),
    Column("publisher", String, nullable=True),
    Column("publication_date", String, nullable=True),
    Column("version", String, default="1.0"),
    Column("final_quality_score", Float, nullable=True),
    Column("validation_passed", Boolean, default=False),
    Column("is_public", Boolean, default=False),
    Column("download_count", Integer, default=0),
    Column("created_at", String, nullable=False),
)


async def create_tables():
    """Create all database tables"""
    try:
        # Ensure data directory exists
        os.makedirs("./data", exist_ok=True)
        
        # Extract the database path from DATABASE_URL
        # Format: sqlite+aiosqlite:///./data/agentic_rag.db
        if "sqlite" in DATABASE_URL:
            db_path = DATABASE_URL.split("///")[-1]
            logger.info(f"Database path: {db_path}")
        
        # Create engine for table creation (synchronous)
        sync_database_url = DATABASE_URL.replace("+aiosqlite", "")
        engine = create_engine(sync_database_url)
        
        # Create all tables
        metadata.create_all(engine)
        logger.info("✅ Database tables created successfully")
        
        # Connect to database (asynchronous)
        await database.connect()
        logger.info("✅ Connected to database")
        
        # Verify tables exist
        await verify_tables()
        
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {str(e)}")
        raise


async def verify_tables():
    """Verify that all required tables exist"""
    try:
        # Check if projects table exists by querying it
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name='projects'"
        result = await database.fetch_one(query)
        
        if result:
            logger.info("✅ Projects table verified")
        else:
            logger.error("❌ Projects table not found")
            raise Exception("Projects table does not exist")
        
        # List all tables
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = await database.fetch_all(query)
        table_names = [table['name'] for table in tables]
        logger.info(f"📋 Available tables: {', '.join(table_names)}")
        
    except Exception as e:
        logger.error(f"❌ Table verification failed: {str(e)}")
        raise


async def drop_all_tables():
    """Drop all tables (use with caution!)"""
    try:
        sync_database_url = DATABASE_URL.replace("+aiosqlite", "")
        engine = create_engine(sync_database_url)
        metadata.drop_all(engine)
        logger.info("✅ All tables dropped")
    except Exception as e:
        logger.error(f"❌ Failed to drop tables: {str(e)}")
        raise


async def database_health_check():
    """Check database connection health"""
    try:
        # Try to execute a simple query
        query = "SELECT 1"
        await database.fetch_one(query)
        
        return {
            "status": "healthy",
            "database_url": DATABASE_URL.split("///")[-1] if "sqlite" in DATABASE_URL else "configured",
            "connected": database.is_connected
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "connected": False
        }


async def get_db():
    """Dependency for getting database connection"""
    try:
        if not database.is_connected:
            await database.connect()
        yield database
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise


async def close_database():
    """Close database connection"""
    try:
        if database.is_connected:
            await database.disconnect()
            logger.info("✅ Database connection closed")
    except Exception as e:
        logger.error(f"❌ Failed to close database: {str(e)}")


# Migration utilities
async def reset_database():
    """Reset database - Drop and recreate all tables"""
    try:
        logger.warning("⚠️ Resetting database - all data will be lost!")
        
        # Disconnect if connected
        if database.is_connected:
            await database.disconnect()
        
        # Drop all tables
        await drop_all_tables()
        
        # Recreate tables
        await create_tables()
        
        logger.info("✅ Database reset complete")
    except Exception as e:
        logger.error(f"❌ Database reset failed: {str(e)}")
        raise


# Export database instance and utilities
__all__ = [
    "database",
    "create_tables",
    "verify_tables",
    "drop_all_tables",
    "database_health_check",
    "get_db",
    "close_database",
    "reset_database",
    "projects_table",
    "agent_tasks_table",
    "chapters_table",
    "research_data_table",
    "publications_table",
]