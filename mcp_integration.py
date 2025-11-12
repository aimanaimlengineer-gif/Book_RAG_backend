"""
MCP Integration for Agentic RAG Book Generator
Model Context Protocol integration for enhanced agent coordination and tool management
"""
import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import Tool, TextContent, ImageContent, CallToolRequest, CallToolResult
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Fallback types
    class Tool:
        def __init__(self, name: str, description: str, inputSchema: Dict[str, Any]):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema
    
    class TextContent:
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text
    
    class CallToolRequest:
        def __init__(self, name: str, arguments: Dict[str, Any]):
            self.name = name
            self.arguments = arguments
    
    class CallToolResult:
        def __init__(self, content: List[Any], isError: bool = False):
            self.content = content
            self.isError = isError

logger = logging.getLogger(__name__)

class MCPToolType(Enum):
    """MCP Tool Types for the Agentic RAG system"""
    RESEARCH_SEARCH = "research_search"
    CONTENT_GENERATION = "content_generation"
    FACT_CHECKING = "fact_checking"
    IMAGE_GENERATION = "image_generation"
    DOCUMENT_PROCESSING = "document_processing"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    QUALITY_ASSESSMENT = "quality_assessment"
    PUBLICATION_FORMAT = "publication_format"

@dataclass
class MCPAgentTool:
    """MCP tool representation for agent capabilities"""
    name: str
    description: str
    agent_type: str
    parameters: Dict[str, Any]
    capabilities: List[str]
    version: str = "1.0.0"
    
    def to_mcp_tool(self) -> Tool:
        """Convert to MCP Tool format"""
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema={
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys())
            }
        )

class MCPAgentCoordinator:
    """MCP-enhanced agent coordination system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.mcp_enabled = MCP_AVAILABLE and self.config.get("mcp_enabled", True)
        self.tools_registry: Dict[str, MCPAgentTool] = {}
        self.active_sessions: Dict[str, Any] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Initialize MCP tools for each agent
        self._initialize_agent_tools()
        
        logger.info(f"MCP Agent Coordinator initialized (MCP Available: {MCP_AVAILABLE})")
    
    def _initialize_agent_tools(self):
        """Initialize MCP tools for all agents"""
        
        # Category Selection Agent Tools
        self.register_tool(MCPAgentTool(
            name="select_book_category",
            description="Analyze book requirements and select the most appropriate category",
            agent_type="category_selection",
            parameters={
                "title": {"type": "string", "description": "Book title"},
                "target_audience": {"type": "string", "description": "Target audience description"},
                "user_preferences": {"type": "object", "description": "User preferences and requirements"},
                "custom_requirements": {"type": "string", "description": "Custom requirements"}
            },
            capabilities=["category_analysis", "audience_matching", "preference_evaluation"]
        ))
        
        # Research Planning Agent Tools
        self.register_tool(MCPAgentTool(
            name="create_research_plan",
            description="Generate comprehensive research plan and chapter outline",
            agent_type="research_planning",
            parameters={
                "title": {"type": "string", "description": "Book title"},
                "category": {"type": "string", "description": "Book category"},
                "chapter_count": {"type": "integer", "description": "Target number of chapters"},
                "target_length": {"type": "integer", "description": "Target word count"},
                "category_analysis": {"type": "object", "description": "Category analysis results"}
            },
            capabilities=["research_planning", "outline_generation", "content_structuring"]
        ))
        
        # Knowledge Acquisition Agent Tools
        self.register_tool(MCPAgentTool(
            name="acquire_knowledge",
            description="Gather and curate information from multiple sources",
            agent_type="knowledge_acquisition",
            parameters={
                "research_plan": {"type": "object", "description": "Research plan with topics and requirements"},
                "chapter_outline": {"type": "array", "description": "Chapter outline structure"},
                "search_terms": {"type": "array", "description": "Search terms for content discovery"},
                "source_types": {"type": "array", "description": "Types of sources to search"}
            },
            capabilities=["web_scraping", "content_extraction", "source_evaluation", "knowledge_curation"]
        ))
        
        # Fact Checking Agent Tools  
        self.register_tool(MCPAgentTool(
            name="verify_content_accuracy",
            description="Verify accuracy and credibility of research content",
            agent_type="fact_checking",
            parameters={
                "research_data": {"type": "object", "description": "Research data to verify"},
                "category": {"type": "string", "description": "Content category for context"},
                "verification_level": {"type": "string", "description": "Level of verification required"}
            },
            capabilities=["fact_verification", "source_credibility", "cross_referencing", "bias_detection"]
        ))
        
        # Content Generation Agent Tools
        self.register_tool(MCPAgentTool(
            name="generate_book_content",
            description="Generate high-quality book content from research",
            agent_type="content_generation",
            parameters={
                "verified_research": {"type": "object", "description": "Fact-checked research data"},
                "chapter_outline": {"type": "array", "description": "Detailed chapter structure"},
                "writing_style": {"type": "string", "description": "Target writing style"},
                "target_audience": {"type": "string", "description": "Intended audience"}
            },
            capabilities=["content_generation", "chapter_writing", "style_adaptation", "audience_targeting"]
        ))
        
        # Illustration Agent Tools
        self.register_tool(MCPAgentTool(
            name="create_visual_content",
            description="Generate illustrations, diagrams, and visual elements",
            agent_type="illustration",
            parameters={
                "content": {"type": "object", "description": "Book content requiring visuals"},
                "category": {"type": "string", "description": "Book category for style guidance"},
                "visual_requirements": {"type": "object", "description": "Specific visual requirements"}
            },
            capabilities=["image_generation", "diagram_creation", "visual_design", "illustration_planning"]
        ))
        
        # Editing QA Agent Tools
        self.register_tool(MCPAgentTool(
            name="edit_and_review_content",
            description="Edit, review, and ensure quality of book content",
            agent_type="editing_qa",
            parameters={
                "content": {"type": "object", "description": "Content to edit and review"},
                "writing_style": {"type": "string", "description": "Target writing style"},
                "quality_standards": {"type": "object", "description": "Quality requirements"}
            },
            capabilities=["content_editing", "quality_assessment", "style_consistency", "grammar_checking"]
        ))
        
        # Publication Agent Tools
        self.register_tool(MCPAgentTool(
            name="create_publication",
            description="Format and publish book in multiple formats",
            agent_type="publication",
            parameters={
                "final_content": {"type": "object", "description": "Final edited content"},
                "output_formats": {"type": "array", "description": "Desired output formats"},
                "publication_metadata": {"type": "object", "description": "Publication metadata"}
            },
            capabilities=["format_conversion", "layout_design", "publication_generation", "multi_format_output"]
        ))
    
    def register_tool(self, tool: MCPAgentTool):
        """Register an MCP tool"""
        self.tools_registry[tool.name] = tool
        
        # Update agent capabilities
        if tool.agent_type not in self.agent_capabilities:
            self.agent_capabilities[tool.agent_type] = []
        
        self.agent_capabilities[tool.agent_type].extend(tool.capabilities)
        
        logger.debug(f"Registered MCP tool: {tool.name} for agent: {tool.agent_type}")
    
    def get_available_tools(self) -> List[Tool]:
        """Get all available MCP tools"""
        return [tool.to_mcp_tool() for tool in self.tools_registry.values()]
    
    def get_tools_for_agent(self, agent_type: str) -> List[Tool]:
        """Get MCP tools for specific agent type"""
        return [
            tool.to_mcp_tool() 
            for tool in self.tools_registry.values() 
            if tool.agent_type == agent_type
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Execute an MCP tool"""
        
        if tool_name not in self.tools_registry:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Tool {tool_name} not found")],
                isError=True
            )
        
        tool = self.tools_registry[tool_name]
        
        try:
            # Route to appropriate agent
            result = await self._route_tool_execution(tool, arguments)
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, default=str))],
                isError=False
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {str(e)}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Tool execution error: {str(e)}")],
                isError=True
            )
    
    async def _route_tool_execution(self, tool: MCPAgentTool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route tool execution to appropriate agent"""
        
        # Import agents dynamically to avoid circular imports
        if tool.agent_type == "category_selection":
            from agents_category_selection import CategorySelectionAgent
            from services_llm_service import LLMService
            
            llm_service = LLMService(self.config)
            agent = CategorySelectionAgent(llm_service, self.config)
            
        elif tool.agent_type == "research_planning":
            from agents_research_planning import ResearchPlanningAgent
            from services_llm_service import LLMService
            from services_web_scraping_service import WebScrapingService
            
            llm_service = LLMService(self.config)
            web_service = WebScrapingService(self.config)
            agent = ResearchPlanningAgent(llm_service, web_service, self.config)
            
        elif tool.agent_type == "knowledge_acquisition":
            from agents_knowledge_acquisition import KnowledgeAcquisitionAgent
            from services_llm_service import LLMService  
            from services_web_scraping_service import WebScrapingService
            from services_vector_db_service import VectorDBService
            
            llm_service = LLMService(self.config)
            web_service = WebScrapingService(self.config)
            vector_service = VectorDBService(self.config)
            agent = KnowledgeAcquisitionAgent(llm_service, web_service, vector_service, self.config)
            
        elif tool.agent_type == "fact_checking":
            from agents_fact_checking import FactCheckingAgent
            from services_llm_service import LLMService
            from services_web_scraping_service import WebScrapingService
            
            llm_service = LLMService(self.config)
            web_service = WebScrapingService(self.config)
            agent = FactCheckingAgent(llm_service, web_service, self.config)
            
        elif tool.agent_type == "content_generation":
            from agents_content_generation import ContentGenerationAgent
            from services_llm_service import LLMService
            from services_embedding_service import EmbeddingService
            
            llm_service = LLMService(self.config)
            embedding_service = EmbeddingService(self.config)
            agent = ContentGenerationAgent(llm_service, embedding_service, self.config)
            
        elif tool.agent_type == "illustration":
            from agents_illustration import IllustrationAgent
            from services_llm_service import LLMService
            
            llm_service = LLMService(self.config)
            agent = IllustrationAgent(llm_service, None, self.config)
            
        elif tool.agent_type == "editing_qa":
            from agents_editing_qa import EditingQAAgent
            from services_llm_service import LLMService
            
            llm_service = LLMService(self.config)
            agent = EditingQAAgent(llm_service, self.config)
            
        elif tool.agent_type == "publication":
            from agents_publication import PublicationAgent
            from services_llm_service import LLMService
            
            llm_service = LLMService(self.config)
            agent = PublicationAgent(llm_service, self.config)
            
        else:
            raise ValueError(f"Unknown agent type: {tool.agent_type}")
        
        # Create task and execute
        from agents_base_agent import BaseAgentTask
        
        task = BaseAgentTask(
            book_project_id=arguments.get("project_id", "mcp_task"),
            agent_type=tool.agent_type,
            task_name=tool.name,
            input_data=arguments,
            priority="normal",
            timeout=300
        )
        
        result = await agent.execute_task(task)
        
        return {
            "success": result.success,
            "output_data": result.output_data,
            "quality_score": result.quality_score,
            "execution_time": result.execution_time,
            "metadata": result.metadata,
            "error_message": result.error_message
        }
    
    async def create_mcp_server(self, server_name: str = "agentic-rag-server") -> Optional[Any]:
        """Create MCP server for external tool access"""
        
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, cannot create server")
            return None
        
        try:
            # This would set up an MCP server
            # Implementation depends on specific MCP server framework
            logger.info(f"MCP server '{server_name}' would be created here")
            
            # Return server instance
            return {"server_name": server_name, "tools": len(self.tools_registry)}
            
        except Exception as e:
            logger.error(f"Failed to create MCP server: {str(e)}")
            return None
    
    async def connect_to_mcp_server(self, server_params: Dict[str, Any]) -> bool:
        """Connect to external MCP server"""
        
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, cannot connect to server")
            return False
        
        try:
            server_name = server_params.get("name", "external-server")
            
            # This would establish connection to external MCP server
            logger.info(f"Would connect to MCP server: {server_name}")
            
            self.active_sessions[server_name] = {
                "connected_at": datetime.utcnow(),
                "server_params": server_params,
                "status": "connected"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            return False
    
    def get_tool_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive tool capabilities report"""
        
        capabilities_report = {
            "total_tools": len(self.tools_registry),
            "mcp_enabled": self.mcp_enabled,
            "mcp_available": MCP_AVAILABLE,
            "agent_types": list(self.agent_capabilities.keys()),
            "tools_by_agent": {},
            "capabilities_by_agent": self.agent_capabilities,
            "tool_details": []
        }
        
        # Group tools by agent
        for agent_type in self.agent_capabilities.keys():
            agent_tools = [
                tool.name for tool in self.tools_registry.values() 
                if tool.agent_type == agent_type
            ]
            capabilities_report["tools_by_agent"][agent_type] = agent_tools
        
        # Add tool details
        for tool in self.tools_registry.values():
            capabilities_report["tool_details"].append({
                "name": tool.name,
                "description": tool.description,
                "agent_type": tool.agent_type,
                "capabilities": tool.capabilities,
                "parameters": list(tool.parameters.keys())
            })
        
        return capabilities_report
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of MCP system"""
        
        health_status = {
            "mcp_available": MCP_AVAILABLE,
            "mcp_enabled": self.mcp_enabled,
            "registered_tools": len(self.tools_registry),
            "active_sessions": len(self.active_sessions),
            "agent_types": len(self.agent_capabilities),
            "timestamp": datetime.utcnow(),
            "status": "healthy"
        }
        
        # Test tool execution
        try:
            test_result = await self.execute_tool(
                "select_book_category",
                {
                    "title": "MCP Health Check",
                    "target_audience": "developers",
                    "user_preferences": {"test": True},
                    "custom_requirements": "Health check test"
                }
            )
            
            if not test_result.isError:
                health_status["tool_execution_test"] = "passed"
            else:
                health_status["tool_execution_test"] = "failed"
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["tool_execution_test"] = "failed"
            health_status["error"] = str(e)
            health_status["status"] = "unhealthy"
        
        return health_status

class MCPWorkflowOrchestrator:
    """MCP-enhanced workflow orchestration"""
    
    def __init__(self, mcp_coordinator: MCPAgentCoordinator):
        self.mcp_coordinator = mcp_coordinator
        self.workflow_stages = [
            "category_selection",
            "research_planning", 
            "knowledge_acquisition",
            "fact_checking",
            "content_generation",
            "illustration",
            "editing_qa",
            "publication"
        ]
    
    async def execute_book_generation_workflow(self, initial_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete book generation workflow using MCP tools"""
        
        workflow_results = {
            "project_id": initial_request.get("project_id", f"mcp_workflow_{datetime.utcnow().timestamp()}"),
            "started_at": datetime.utcnow().isoformat(),
            "stages": {},
            "overall_success": False,
            "final_output": None
        }
        
        current_data = initial_request.copy()
        
        try:
            # Stage 1: Category Selection
            logger.info("MCP Workflow: Starting category selection")
            category_result = await self.mcp_coordinator.execute_tool(
                "select_book_category",
                {
                    "title": current_data.get("title"),
                    "target_audience": current_data.get("target_audience"),
                    "user_preferences": current_data.get("user_preferences", {}),
                    "custom_requirements": current_data.get("custom_requirements", "")
                }
            )
            
            if category_result.isError:
                raise Exception(f"Category selection failed: {category_result.content[0].text}")
            
            category_data = json.loads(category_result.content[0].text)
            workflow_results["stages"]["category_selection"] = category_data
            current_data.update(category_data["output_data"])
            
            # Stage 2: Research Planning
            logger.info("MCP Workflow: Starting research planning")
            research_result = await self.mcp_coordinator.execute_tool(
                "create_research_plan",
                {
                    "title": current_data.get("title"),
                    "category": current_data.get("selected_category"),
                    "chapter_count": current_data.get("chapter_count", 8),
                    "target_length": current_data.get("target_length", 15000),
                    "category_analysis": current_data.get("category_analysis", {})
                }
            )
            
            if research_result.isError:
                raise Exception(f"Research planning failed: {research_result.content[0].text}")
            
            research_data = json.loads(research_result.content[0].text)
            workflow_results["stages"]["research_planning"] = research_data
            current_data.update(research_data["output_data"])
            
            # Stage 3: Knowledge Acquisition
            logger.info("MCP Workflow: Starting knowledge acquisition")
            knowledge_result = await self.mcp_coordinator.execute_tool(
                "acquire_knowledge",
                {
                    "research_plan": current_data.get("research_plan"),
                    "chapter_outline": current_data.get("chapter_outline"),
                    "search_terms": current_data.get("search_terms", []),
                    "source_types": ["articles", "reports", "guides"]
                }
            )
            
            if knowledge_result.isError:
                raise Exception(f"Knowledge acquisition failed: {knowledge_result.content[0].text}")
            
            knowledge_data = json.loads(knowledge_result.content[0].text)
            workflow_results["stages"]["knowledge_acquisition"] = knowledge_data
            current_data.update(knowledge_data["output_data"])
            
            # Stage 4: Fact Checking
            logger.info("MCP Workflow: Starting fact checking")
            fact_check_result = await self.mcp_coordinator.execute_tool(
                "verify_content_accuracy",
                {
                    "research_data": current_data.get("knowledge_base"),
                    "category": current_data.get("selected_category"),
                    "verification_level": "standard"
                }
            )
            
            if fact_check_result.isError:
                raise Exception(f"Fact checking failed: {fact_check_result.content[0].text}")
            
            fact_check_data = json.loads(fact_check_result.content[0].text)
            workflow_results["stages"]["fact_checking"] = fact_check_data
            current_data.update(fact_check_data["output_data"])
            
            # Stage 5: Content Generation
            logger.info("MCP Workflow: Starting content generation")
            content_result = await self.mcp_coordinator.execute_tool(
                "generate_book_content",
                {
                    "verified_research": current_data.get("verified_research"),
                    "chapter_outline": current_data.get("chapter_outline"),
                    "writing_style": current_data.get("writing_style", "informative"),
                    "target_audience": current_data.get("target_audience")
                }
            )
            
            if content_result.isError:
                raise Exception(f"Content generation failed: {content_result.content[0].text}")
            
            content_data = json.loads(content_result.content[0].text)
            workflow_results["stages"]["content_generation"] = content_data
            current_data.update(content_data["output_data"])
            
            # Stage 6: Illustration (Optional)
            if current_data.get("include_illustrations", True):
                logger.info("MCP Workflow: Starting illustration generation")
                illustration_result = await self.mcp_coordinator.execute_tool(
                    "create_visual_content",
                    {
                        "content": current_data.get("chapters"),
                        "category": current_data.get("selected_category"),
                        "visual_requirements": current_data.get("visual_requirements", {})
                    }
                )
                
                if not illustration_result.isError:
                    illustration_data = json.loads(illustration_result.content[0].text)
                    workflow_results["stages"]["illustration"] = illustration_data
                    current_data.update(illustration_data["output_data"])
            
            # Stage 7: Editing & QA
            logger.info("MCP Workflow: Starting editing and QA")
            editing_result = await self.mcp_coordinator.execute_tool(
                "edit_and_review_content",
                {
                    "content": {
                        "introduction": current_data.get("introduction"),
                        "chapters": current_data.get("chapters"),
                        "conclusion": current_data.get("conclusion")
                    },
                    "writing_style": current_data.get("writing_style"),
                    "quality_standards": current_data.get("quality_standards", {})
                }
            )
            
            if editing_result.isError:
                raise Exception(f"Editing failed: {editing_result.content[0].text}")
            
            editing_data = json.loads(editing_result.content[0].text)
            workflow_results["stages"]["editing_qa"] = editing_data
            current_data.update(editing_data["output_data"])
            
            # Stage 8: Publication
            logger.info("MCP Workflow: Starting publication")
            publication_result = await self.mcp_coordinator.execute_tool(
                "create_publication",
                {
                    "final_content": current_data.get("edited_content"),
                    "output_formats": current_data.get("output_formats", ["pdf", "epub"]),
                    "publication_metadata": {
                        "title": current_data.get("title"),
                        "author": current_data.get("author", "AI Generated"),
                        "category": current_data.get("selected_category")
                    }
                }
            )
            
            if publication_result.isError:
                raise Exception(f"Publication failed: {publication_result.content[0].text}")
            
            publication_data = json.loads(publication_result.content[0].text)
            workflow_results["stages"]["publication"] = publication_data
            
            # Success
            workflow_results["overall_success"] = True
            workflow_results["final_output"] = publication_data["output_data"]
            workflow_results["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info("MCP Workflow: Book generation completed successfully")
            
        except Exception as e:
            logger.error(f"MCP Workflow failed: {str(e)}")
            workflow_results["error"] = str(e)
            workflow_results["failed_at"] = datetime.utcnow().isoformat()
        
        return workflow_results

# MCP FastAPI Integration
class MCPFastAPIIntegration:
    """FastAPI integration for MCP tools"""
    
    def __init__(self, app, mcp_coordinator: MCPAgentCoordinator):
        self.app = app
        self.mcp_coordinator = mcp_coordinator
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup MCP-related API routes"""
        
        @self.app.get("/mcp/tools")
        async def get_mcp_tools():
            """Get available MCP tools"""
            capabilities = self.mcp_coordinator.get_tool_capabilities()
            return {"tools": capabilities}
        
        @self.app.post("/mcp/execute/{tool_name}")
        async def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]):
            """Execute MCP tool"""
            result = await self.mcp_coordinator.execute_tool(tool_name, arguments)
            
            return {
                "tool_name": tool_name,
                "success": not result.isError,
                "result": result.content[0].text if result.content else None,
                "executed_at": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/mcp/health")
        async def mcp_health_check():
            """MCP system health check"""
            return await self.mcp_coordinator.health_check()
        
        @self.app.post("/mcp/workflow/book-generation")
        async def execute_mcp_book_workflow(request_data: Dict[str, Any]):
            """Execute complete book generation workflow via MCP"""
            orchestrator = MCPWorkflowOrchestrator(self.mcp_coordinator)
            result = await orchestrator.execute_book_generation_workflow(request_data)
            return result

def create_mcp_integration(config: Dict[str, Any]) -> MCPAgentCoordinator:
    """Factory function to create MCP integration"""
    
    return MCPAgentCoordinator(config)

# Usage example
if __name__ == "__main__":
    # Example usage
    config = {
        "mcp_enabled": True,
        "groq_api_key": "your-groq-key",
        "openai_api_key": "your-openai-key"
    }
    
    # Create MCP coordinator
    mcp_coordinator = create_mcp_integration(config)
    
    # Example tool execution
    async def example_usage():
        result = await mcp_coordinator.execute_tool(
            "select_book_category",
            {
                "title": "Python Programming Guide", 
                "target_audience": "beginners",
                "user_preferences": {"focus": "practical"},
                "custom_requirements": "Include examples"
            }
        )
        
        print("MCP Tool Result:", result.content[0].text if result.content else "No result")
    
    # Run example
    # asyncio.run(example_usage())