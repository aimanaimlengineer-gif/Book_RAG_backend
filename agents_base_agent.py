"""
Base Agent Classes - Foundation for all specialized agents
"""
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ==========================================
# Enums and Data Classes
# ==========================================

class AgentCapability(str, Enum):
    """Agent capabilities enumeration"""
    # Analysis & Planning
    CATEGORY_ANALYSIS = "category_analysis"
    RESEARCH_PLANNING = "research_planning"
    
    # Data & Content
    CONTENT_EXTRACTION = "content_extraction"
    DATA_COLLECTION = "data_collection"
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    WEB_SCRAPING = "web_scraping"
    
    # Verification & Checking
    FACT_CHECKING = "fact_checking"
    FACT_VERIFICATION = "fact_verification"
    QUALITY_ASSURANCE = "quality_assurance"
    
    # Content Creation
    CONTENT_GENERATION = "content_generation"
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_WRITING = "technical_writing"
    
    # Editing & Quality
    EDITING_QA = "editing_qa"
    CONTENT_EDITING = "content_editing"
    PROOFREADING = "proofreading"
    
    # Publishing
    PUBLICATION = "publication"
    FORMAT_CONVERSION = "format_conversion"


@dataclass
class BaseAgentTask:
    """Base task structure for agents"""
    book_project_id: str
    agent_type: str
    task_name: str
    input_data: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "book_project_id": self.book_project_id,
            "agent_type": self.agent_type,
            "task_name": self.task_name,
            "input_data": self.input_data,
            "config": self.config,
            "priority": self.priority,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class BaseAgentResult:
    """Base result structure from agent execution"""
    success: bool
    task_id: str
    agent_type: str
    execution_time: float
    output_data: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "success": self.success,
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "execution_time": self.execution_time,
            "output_data": self.output_data,
            "quality_score": self.quality_score,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


# ==========================================
# Base Agent Class
# ==========================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Provides common functionality and enforces interface contracts.
    """
    
    def __init__(
        self,
        agent_type: str,
        role: str,
        goal: str,
        backstory: str,
        capabilities: List[AgentCapability],
        llm_service=None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_type = agent_type
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.capabilities = capabilities
        self.llm_service = llm_service
        self.config = config or {}
        
        # Execution tracking
        self.status = "idle"
        self.current_task: Optional[BaseAgentTask] = None
        self.task_history: List[BaseAgentResult] = []
        self.execution_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time: Optional[datetime] = None
        
        logger.info(f"Initialized {agent_type} agent with capabilities: {[c.value for c in capabilities]}")
    
    # ==========================================
    # Main Execution Methods
    # ==========================================
    
    async def execute(self, request: Any) -> Dict[str, Any]:
        """
        Execute method for orchestrator compatibility.
        Converts BookGenerationRequest to BaseAgentTask and executes it.
        
        This is the method called by the orchestrator.
        """
        try:
            logger.info(f"[{self.agent_type}] Execute called with request")
            
            # Create a task from the request
            task = BaseAgentTask(
                book_project_id=getattr(request, 'id', str(uuid.uuid4())),
                agent_type=self.agent_type,
                task_name=f"{self.agent_type}_execution",
                input_data={
                    "title": getattr(request, 'title', ''),
                    "topic": getattr(request, 'topic', ''),
                    "category": getattr(request, 'category', ''),
                    "genre": getattr(request, 'genre', None),
                    "author": getattr(request, 'author', 'AI Generated'),
                    "target_audience": getattr(request, 'target_audience', ''),
                    "chapter_count": getattr(request, 'chapter_count', 8),
                    "target_length": getattr(request, 'target_length', 15000),
                    "writing_style": str(getattr(request, 'writing_style', 'informative')),
                    "user_preferences": getattr(request, 'user_preferences', {}),
                    "custom_requirements": getattr(request, 'custom_requirements', ''),
                    "language": getattr(request, 'language', 'en'),
                    "output_formats": getattr(request, 'output_formats', ['pdf']),
                    "include_illustrations": getattr(request, 'include_illustrations', True),
                },
                config=self.config
            )
            
            # Execute the task
            logger.info(f"[{self.agent_type}] Executing task...")
            result = await self.execute_task(task)
            
            # Return output data as dict
            if result.success:
                logger.info(f"[{self.agent_type}] Execution successful (quality: {result.quality_score:.2f})")
                return {
                    "status": "completed",
                    "agent_type": self.agent_type,
                    "data": result.output_data,
                    "quality_score": result.quality_score,
                    "execution_time": result.execution_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                logger.error(f"[{self.agent_type}] Execution failed: {result.error_message}")
                return {
                    "status": "failed",
                    "agent_type": self.agent_type,
                    "error": result.error_message or "Unknown error",
                    "execution_time": result.execution_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"[{self.agent_type}] Execute failed with exception: {str(e)}", exc_info=True)
            return {
                "status": "failed",
                "agent_type": self.agent_type,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def execute_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """
        Execute a task with full lifecycle management.
        This is called by the execute() method above.
        """
        start_time = datetime.utcnow()
        self.current_task = task
        self.status = "running"
        
        try:
            # Validate capabilities
            required_capabilities = self._get_required_capabilities(task)
            self._validate_capabilities(required_capabilities)
            
            # Pre-execution hook
            await self._pre_execution_hook(task)
            
            # Execute core task logic (implemented by subclasses)
            logger.info(f"[{self.agent_type}] Starting core task execution")
            result = await self._execute_core_task(task)
            
            # Post-execution hook
            await self._post_execution_hook(task, result)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Update tracking
            self._update_execution_tracking(result, execution_time)
            
            # Store in history
            self.task_history.append(result)
            
            logger.info(f"[{self.agent_type}] Task completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"[{self.agent_type}] Task execution failed: {str(e)}", exc_info=True)
            
            error_result = BaseAgentResult(
                success=False,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=execution_time,
                error_message=str(e),
                metadata={"error_type": type(e).__name__}
            )
            
            self._update_execution_tracking(error_result, execution_time)
            self.task_history.append(error_result)
            
            return error_result
            
        finally:
            self.current_task = None
            self.status = "idle"
    
    # ==========================================
    # Abstract Methods (Must be implemented by subclasses)
    # ==========================================
    
    @abstractmethod
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """
        Core task execution logic - must be implemented by each agent.
        
        Args:
            task: The task to execute
            
        Returns:
            BaseAgentResult with execution results
        """
        pass
    
    @abstractmethod
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """
        Get required capabilities for this task - must be implemented by each agent.
        
        Args:
            task: The task to check capabilities for
            
        Returns:
            List of required capabilities
        """
        pass
    
    # ==========================================
    # Lifecycle Hooks
    # ==========================================
    
    async def _pre_execution_hook(self, task: BaseAgentTask):
        """Hook called before task execution"""
        logger.debug(f"[{self.agent_type}] Pre-execution hook")
    
    async def _post_execution_hook(self, task: BaseAgentTask, result: BaseAgentResult):
        """Hook called after task execution"""
        logger.debug(f"[{self.agent_type}] Post-execution hook")
    
    # ==========================================
    # Helper Methods
    # ==========================================
    
    def _validate_capabilities(self, required_capabilities: List[AgentCapability]):
        """Validate that agent has required capabilities"""
        missing = [cap for cap in required_capabilities if cap not in self.capabilities]
        if missing:
            raise ValueError(
                f"Agent {self.agent_type} missing required capabilities: {[c.value for c in missing]}"
            )
    
    def _update_execution_tracking(self, result: BaseAgentResult, execution_time: float):
        """Update execution tracking metrics"""
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.last_execution_time = datetime.utcnow()
        
        if result.success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        avg_execution_time = (
            self.total_execution_time / self.execution_count 
            if self.execution_count > 0 
            else 0.0
        )
        
        success_rate = (
            self.success_count / self.execution_count 
            if self.execution_count > 0 
            else 0.0
        )
        
        return {
            "agent_type": self.agent_type,
            "status": "healthy" if success_rate > 0.8 else "degraded" if success_rate > 0.5 else "unhealthy",
            "current_status": self.status,
            "capabilities": [c.value for c in self.capabilities],
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "current_task": self.current_task.task_id if self.current_task else None
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            "agent_type": self.agent_type,
            "total_executions": self.execution_count,
            "successful_executions": self.success_count,
            "failed_executions": self.error_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": (
                self.total_execution_time / self.execution_count 
                if self.execution_count > 0 
                else 0.0
            ),
            "success_rate": (
                self.success_count / self.execution_count 
                if self.execution_count > 0 
                else 0.0
            ),
            "recent_tasks": [
                {
                    "task_id": result.task_id,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "quality_score": result.quality_score
                }
                for result in self.task_history[-10:]  # Last 10 tasks
            ]
        }


# ==========================================
# Agent Coordinator
# ==========================================

class AgentCoordinator:
    """
    Coordinates multiple agents and manages agent lifecycle.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[BaseAgentTask] = []
        self.completed_tasks: List[BaseAgentResult] = []
        
        logger.info("AgentCoordinator initialized")
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator"""
        self.agents[agent.agent_type] = agent
        logger.info(f"Registered agent: {agent.agent_type} ({agent.role})")
    
    def get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """Get an agent by type"""
        return self.agents.get(agent_type)
    
    def list_agents(self) -> List[str]:
        """List all registered agent types"""
        return list(self.agents.keys())
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get status of all agents in the system"""
        agent_statuses = {}
        
        for agent_type, agent in self.agents.items():
            try:
                agent_statuses[agent_type] = await agent.health_check()
            except Exception as e:
                logger.error(f"Failed to get status for {agent_type}: {str(e)}")
                agent_statuses[agent_type] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.status == "running"),
            "agents": agent_statuses,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def execute_workflow(
        self, 
        workflow: List[str], 
        request: Any
    ) -> List[Dict[str, Any]]:
        """
        Execute a workflow of agent tasks in sequence.
        
        Args:
            workflow: List of agent types to execute in order
            request: The request object to pass to each agent
            
        Returns:
            List of results from each agent
        """
        results = []
        
        for agent_type in workflow:
            agent = self.get_agent(agent_type)
            
            if not agent:
                logger.warning(f"Agent not found in workflow: {agent_type}")
                results.append({
                    "agent_type": agent_type,
                    "status": "failed",
                    "error": "Agent not found"
                })
                continue
            
            try:
                logger.info(f"Executing workflow step: {agent_type}")
                result = await agent.execute(request)
                results.append(result)
                
                # Stop workflow if agent failed
                if result.get("status") != "completed":
                    logger.error(f"Workflow stopped at {agent_type} due to failure")
                    break
                    
            except Exception as e:
                logger.error(f"Workflow execution failed at {agent_type}: {str(e)}")
                results.append({
                    "agent_type": agent_type,
                    "status": "failed",
                    "error": str(e)
                })
                break
        
        return results
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        return {
            agent_type: agent.get_performance_metrics()
            for agent_type, agent in self.agents.items()
        }


# ==========================================
# Exports
# ==========================================

__all__ = [
    "BaseAgent",
    "BaseAgentTask",
    "BaseAgentResult",
    "AgentCapability",
    "AgentCoordinator"
]