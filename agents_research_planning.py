"""
Research & Planning Agent - Creates comprehensive book outlines and research plans
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from agents_base_agent import BaseAgent, BaseAgentTask, BaseAgentResult, AgentCapability
from config_settings import BOOK_CATEGORIES

logger = logging.getLogger(__name__)

class ResearchPlanningAgent(BaseAgent):
    """Agent responsible for research planning and book structure creation"""
    
    def __init__(self, llm_service, web_scraping_service, config: Dict[str, Any] = None):
        self.web_scraping_service = web_scraping_service
        super().__init__(
            agent_type="research_planning",
            role="Research & Planning Strategist",
            goal="Create comprehensive book outlines, chapter structures, and detailed research plans",
            backstory="You are an experienced book planner and research strategist with expertise in content architecture, educational design, and comprehensive research methodology. You excel at breaking down complex topics into structured, engaging chapters.",
            capabilities=[AgentCapability.RESEARCH_PLANNING, AgentCapability.CONTENT_EXTRACTION],
            llm_service=llm_service,
            config=config
        )
    
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """Get required capabilities for research planning tasks"""
        return [AgentCapability.RESEARCH_PLANNING]
    
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """Execute research planning task"""
        try:
            input_data = task.input_data
            
            # Extract task parameters
            title = input_data.get("title", "")
            category = input_data.get("category", "")
            target_audience = input_data.get("target_audience", "")
            chapter_count = input_data.get("chapter_count", 8)
            target_length = input_data.get("target_length", 15000)
            category_analysis = input_data.get("category_analysis", {})
            
            # Create comprehensive research plan
            research_plan = await self._create_research_plan(
                title, category, target_audience, chapter_count, target_length, category_analysis
            )
            
            # Generate detailed chapter outline
            chapter_outline = await self._generate_chapter_outline(
                title, category, chapter_count, research_plan
            )
            
            # Create content structure
            content_structure = await self._design_content_structure(
                research_plan, chapter_outline, target_length
            )
            
            # Generate research requirements
            research_requirements = await self._generate_research_requirements(
                research_plan, chapter_outline
            )
            
            output_data = {
                "research_plan": research_plan,
                "chapter_outline": chapter_outline,
                "content_structure": content_structure,
                "research_requirements": research_requirements,
                "estimated_timeline": self._estimate_timeline(chapter_count, target_length),
                "quality_checkpoints": self._define_quality_checkpoints(chapter_outline)
            }
            
            quality_score = self._calculate_planning_quality_score(output_data, task)
            
            return BaseAgentResult(
                success=True,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=0.0,
                output_data=output_data,
                quality_score=quality_score,
                metadata={
                    "planning_method": "structured_llm_analysis",
                    "chapters_planned": len(chapter_outline),
                    "research_sources_identified": len(research_requirements.get("sources", []))
                }
            )
            
        except Exception as e:
            logger.error(f"Research planning failed: {str(e)}")
            raise
    
    async def _create_research_plan(
        self,
        title: str,
        category: str,
        target_audience: str,
        chapter_count: int,
        target_length: int,
        category_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive research plan"""
        
        category_config = BOOK_CATEGORIES.get(category, {})
        
        planning_prompt = f"""
        Create a comprehensive research plan for a book with the following specifications:
        
        Title: {title}
        Category: {category} ({category_config.get('name', '')})
        Target Audience: {target_audience}
        Chapter Count: {chapter_count}
        Target Length: {target_length} words
        Writing Style: {category_config.get('style', 'informative')}
        
        Category Analysis: {json.dumps(category_analysis, indent=2)}
        
        Create a detailed research plan including:
        1. Core themes and topics to cover
        2. Key concepts and learning objectives
        3. Research methodology and approach
        4. Primary and secondary research areas
        5. Content gaps to address
        6. Competitive analysis requirements
        7. Expert sources to consult
        8. Data and statistics needed
        9. Case studies and examples required
        10. Quality standards and criteria
        
        Provide the response in JSON format with clear structure.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=planning_prompt,
                max_tokens=2000,
                temperature=0.4
            )
            
            research_plan = json.loads(response)
            
            # Enhance with category-specific requirements
            research_plan["category_requirements"] = self._get_category_requirements(category)
            research_plan["audience_considerations"] = self._analyze_audience_needs(target_audience)
            
            return research_plan
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM research planning failed, using template: {str(e)}")
            return self._fallback_research_plan(title, category, target_audience)
    
    async def _generate_chapter_outline(
        self,
        title: str,
        category: str,
        chapter_count: int,
        research_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate detailed chapter outline"""
        
        outline_prompt = f"""
        Based on this research plan, create a detailed chapter outline for "{title}" with {chapter_count} chapters:
        
        Research Plan: {json.dumps(research_plan, indent=2)}
        
        For each chapter, provide:
        1. Chapter number and title
        2. Learning objectives (3-5 objectives)
        3. Key concepts to cover
        4. Estimated word count
        5. Main sections and subsections
        6. Required research areas
        7. Examples and case studies needed
        8. Exercises or actionable content
        9. Connection to other chapters
        10. Success metrics
        
        Ensure logical flow and progression between chapters.
        Provide response as JSON array of chapter objects.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=outline_prompt,
                max_tokens=2500,
                temperature=0.3
            )
            
            chapter_outline = json.loads(response)
            
            # Validate and enhance outline
            chapter_outline = self._validate_chapter_outline(chapter_outline, chapter_count)
            
            return chapter_outline
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Chapter outline generation failed, using template: {str(e)}")
            return self._fallback_chapter_outline(title, category, chapter_count)
    
    async def _design_content_structure(
        self,
        research_plan: Dict[str, Any],
        chapter_outline: List[Dict[str, Any]],
        target_length: int
    ) -> Dict[str, Any]:
        """Design overall content structure and flow"""
        
        total_chapters = len(chapter_outline)
        avg_chapter_length = target_length // total_chapters
        
        structure = {
            "book_structure": {
                "introduction": {
                    "word_count": int(avg_chapter_length * 0.8),
                    "purpose": "Hook reader, set expectations, provide roadmap"
                },
                "main_content": {
                    "chapters": total_chapters,
                    "avg_chapter_length": avg_chapter_length,
                    "total_word_count": target_length
                },
                "conclusion": {
                    "word_count": int(avg_chapter_length * 0.6),
                    "purpose": "Summarize key points, provide next steps"
                }
            },
            "content_flow": self._design_content_flow(chapter_outline),
            "cross_references": self._identify_cross_references(chapter_outline),
            "appendices": self._plan_appendices(research_plan),
            "visual_elements": self._plan_visual_elements(chapter_outline)
        }
        
        return structure
    
    async def _generate_research_requirements(
        self,
        research_plan: Dict[str, Any],
        chapter_outline: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate specific research requirements"""
        
        requirements = {
            "primary_sources": [],
            "secondary_sources": [],
            "data_requirements": [],
            "expert_interviews": [],
            "case_studies": [],
            "statistics_needed": [],
            "verification_sources": [],
            "research_timeline": {}
        }
        
        # Extract requirements from research plan and chapters
        for chapter in chapter_outline:
            chapter_requirements = chapter.get("required_research_areas", [])
            
            for requirement in chapter_requirements:
                if "statistics" in requirement.lower() or "data" in requirement.lower():
                    requirements["statistics_needed"].append({
                        "chapter": chapter.get("chapter_number"),
                        "requirement": requirement,
                        "priority": "high"
                    })
                elif "expert" in requirement.lower() or "interview" in requirement.lower():
                    requirements["expert_interviews"].append({
                        "chapter": chapter.get("chapter_number"),
                        "expertise_needed": requirement,
                        "priority": "medium"
                    })
                elif "case study" in requirement.lower():
                    requirements["case_studies"].append({
                        "chapter": chapter.get("chapter_number"),
                        "case_study_type": requirement,
                        "priority": "high"
                    })
        
        # Add web research requirements
        if self.web_scraping_service:
            web_sources = await self._identify_web_sources(research_plan)
            requirements["web_sources"] = web_sources
        
        return requirements
    
    def _get_category_requirements(self, category: str) -> Dict[str, Any]:
        """Get specific requirements for book category"""
        category_requirements = {
            "technology": {
                "code_examples": True,
                "technical_accuracy": "critical",
                "current_trends": True,
                "hands_on_exercises": True,
                "tool_recommendations": True
            },
            "business": {
                "case_studies": True,
                "financial_data": True,
                "market_research": True,
                "actionable_strategies": True,
                "roi_metrics": True
            },
            "health": {
                "medical_accuracy": "critical",
                "scientific_citations": True,
                "safety_warnings": True,
                "professional_review": True,
                "evidence_based": True
            },
            "finance": {
                "regulatory_compliance": True,
                "risk_warnings": True,
                "current_regulations": True,
                "practical_examples": True,
                "calculation_methods": True
            },
            "education": {
                "learning_objectives": True,
                "assessment_methods": True,
                "progressive_difficulty": True,
                "practical_applications": True,
                "reference_materials": True
            },
            "lifestyle": {
                "personal_anecdotes": True,
                "practical_tips": True,
                "real_world_examples": True,
                "step_by_step_guides": True,
                "motivational_content": True
            }
        }
        
        return category_requirements.get(category, {})
    
    def _analyze_audience_needs(self, target_audience: str) -> Dict[str, Any]:
        """Analyze target audience needs and preferences"""
        audience_lower = target_audience.lower()
        
        considerations = {
            "knowledge_level": "intermediate",
            "preferred_learning_style": "mixed",
            "attention_span": "medium",
            "practical_focus": True,
            "theoretical_depth": "moderate"
        }
        
        # Adjust based on audience indicators
        if any(word in audience_lower for word in ["beginner", "new", "starter", "novice"]):
            considerations["knowledge_level"] = "beginner"
            considerations["theoretical_depth"] = "basic"
            considerations["step_by_step"] = True
        elif any(word in audience_lower for word in ["expert", "advanced", "professional"]):
            considerations["knowledge_level"] = "advanced"
            considerations["theoretical_depth"] = "deep"
            considerations["technical_detail"] = "high"
        
        if any(word in audience_lower for word in ["student", "academic"]):
            considerations["preferred_learning_style"] = "structured"
            considerations["assessment_needed"] = True
        
        return considerations
    
    def _validate_chapter_outline(
        self,
        outline: List[Dict[str, Any]],
        expected_count: int
    ) -> List[Dict[str, Any]]:
        """Validate and fix chapter outline"""
        
        # Ensure correct number of chapters
        if len(outline) != expected_count:
            # Adjust outline to match expected count
            if len(outline) < expected_count:
                # Add missing chapters
                for i in range(len(outline), expected_count):
                    outline.append({
                        "chapter_number": i + 1,
                        "title": f"Chapter {i + 1}",
                        "learning_objectives": [],
                        "key_concepts": [],
                        "estimated_word_count": 2000,
                        "main_sections": []
                    })
            else:
                # Trim excess chapters
                outline = outline[:expected_count]
        
        # Ensure all chapters have required fields
        for i, chapter in enumerate(outline):
            chapter["chapter_number"] = i + 1
            if "title" not in chapter:
                chapter["title"] = f"Chapter {i + 1}"
            if "learning_objectives" not in chapter:
                chapter["learning_objectives"] = []
            if "estimated_word_count" not in chapter:
                chapter["estimated_word_count"] = 2000
        
        return outline
    
    def _design_content_flow(self, chapter_outline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Design logical content flow between chapters"""
        
        flow = {
            "progression_type": "linear",
            "dependencies": [],
            "reinforcement_points": [],
            "knowledge_building": []
        }
        
        for i, chapter in enumerate(chapter_outline):
            if i > 0:
                flow["dependencies"].append({
                    "chapter": i + 1,
                    "depends_on": [i],
                    "connection_type": "sequential"
                })
        
        return flow
    
    def _identify_cross_references(self, chapter_outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify cross-references between chapters"""
        
        references = []
        
        # Simple keyword-based cross-reference identification
        for i, chapter in enumerate(chapter_outline):
            concepts = chapter.get("key_concepts", [])
            
            for j, other_chapter in enumerate(chapter_outline):
                if i != j:
                    other_concepts = other_chapter.get("key_concepts", [])
                    
                    # Find common concepts
                    common_concepts = [c for c in concepts if c in other_concepts]
                    
                    if common_concepts:
                        references.append({
                            "from_chapter": i + 1,
                            "to_chapter": j + 1,
                            "common_concepts": common_concepts,
                            "reference_type": "concept_link"
                        })
        
        return references
    
    def _plan_appendices(self, research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan appendices and supplementary materials"""
        
        appendices = [
            {
                "title": "Resources and References",
                "content_type": "bibliography",
                "priority": "high"
            },
            {
                "title": "Glossary of Terms",
                "content_type": "definitions",
                "priority": "medium"
            },
            {
                "title": "Additional Reading",
                "content_type": "recommendations",
                "priority": "low"
            }
        ]
        
        return appendices
    
    def _plan_visual_elements(self, chapter_outline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan visual elements and illustrations"""
        
        visual_plan = {
            "diagrams_needed": [],
            "charts_and_graphs": [],
            "illustrations": [],
            "infographics": [],
            "screenshots": []
        }
        
        for chapter in chapter_outline:
            chapter_num = chapter.get("chapter_number", 1)
            
            # Suggest visual elements based on content
            if "process" in str(chapter.get("key_concepts", [])).lower():
                visual_plan["diagrams_needed"].append({
                    "chapter": chapter_num,
                    "type": "process_diagram",
                    "description": f"Process flow for Chapter {chapter_num}"
                })
            
            if "data" in str(chapter.get("key_concepts", [])).lower():
                visual_plan["charts_and_graphs"].append({
                    "chapter": chapter_num,
                    "type": "data_visualization",
                    "description": f"Data charts for Chapter {chapter_num}"
                })
        
        return visual_plan
    
    def _estimate_timeline(self, chapter_count: int, target_length: int) -> Dict[str, Any]:
        """Estimate project timeline"""
        
        # Base estimates (in hours)
        research_time_per_chapter = 4
        writing_time_per_1000_words = 3
        editing_time_per_chapter = 2
        
        total_research_time = chapter_count * research_time_per_chapter
        total_writing_time = (target_length / 1000) * writing_time_per_1000_words
        total_editing_time = chapter_count * editing_time_per_chapter
        
        timeline = {
            "research_phase": f"{total_research_time} hours",
            "writing_phase": f"{total_writing_time} hours",
            "editing_phase": f"{total_editing_time} hours",
            "total_estimated_time": f"{total_research_time + total_writing_time + total_editing_time} hours",
            "estimated_days": f"{int((total_research_time + total_writing_time + total_editing_time) / 8)} working days"
        }
        
        return timeline
    
    def _define_quality_checkpoints(self, chapter_outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Define quality checkpoints throughout the project"""
        
        checkpoints = []
        
        # Research completion checkpoint
        checkpoints.append({
            "phase": "research_completion",
            "criteria": [
                "All primary sources identified",
                "Key statistics and data collected",
                "Expert interviews completed",
                "Case studies documented"
            ],
            "milestone": "25% completion"
        })
        
        # First draft checkpoint
        checkpoints.append({
            "phase": "first_draft",
            "criteria": [
                "All chapters drafted",
                "Word count targets met",
                "Chapter flow established",
                "Key concepts covered"
            ],
            "milestone": "60% completion"
        })
        
        # Quality review checkpoint
        checkpoints.append({
            "phase": "quality_review",
            "criteria": [
                "Fact-checking completed",
                "Content accuracy verified",
                "Style consistency achieved",
                "Learning objectives met"
            ],
            "milestone": "85% completion"
        })
        
        return checkpoints
    
    async def _identify_web_sources(self, research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify web sources for research"""
        
        # This would integrate with the web scraping service
        web_sources = [
            {
                "type": "industry_reports",
                "priority": "high",
                "search_terms": research_plan.get("core_themes", [])
            },
            {
                "type": "academic_papers",
                "priority": "high",
                "search_terms": research_plan.get("key_concepts", [])
            },
            {
                "type": "expert_blogs",
                "priority": "medium",
                "search_terms": research_plan.get("core_themes", [])
            }
        ]
        
        return web_sources
    
    def _fallback_research_plan(self, title: str, category: str, target_audience: str) -> Dict[str, Any]:
        """Fallback research plan when LLM fails"""
        
        return {
            "core_themes": [f"Introduction to {category}", f"Fundamentals", f"Advanced concepts", f"Practical applications"],
            "key_concepts": [f"{category} basics", "Best practices", "Common challenges", "Solutions"],
            "research_methodology": "Mixed methods approach with primary and secondary sources",
            "quality_standards": "High accuracy with multiple source verification"
        }
    
    def _fallback_chapter_outline(self, title: str, category: str, chapter_count: int) -> List[Dict[str, Any]]:
        """Fallback chapter outline when LLM fails"""
        
        outline = []
        
        for i in range(chapter_count):
            outline.append({
                "chapter_number": i + 1,
                "title": f"Chapter {i + 1}: {category.title()} Topic {i + 1}",
                "learning_objectives": [f"Understand key concept {i + 1}", f"Apply {category} principles"],
                "key_concepts": [f"Concept {i + 1}", f"{category.title()} principle"],
                "estimated_word_count": 2000,
                "main_sections": [f"Introduction", f"Core concepts", f"Examples", f"Summary"]
            })
        
        return outline
    
    def _calculate_planning_quality_score(
        self,
        output_data: Dict[str, Any],
        task: BaseAgentTask
    ) -> float:
        """Calculate quality score for research planning"""
        
        base_score = 0.7
        
        # Check completeness of research plan
        research_plan = output_data.get("research_plan", {})
        if len(research_plan.get("core_themes", [])) >= 3:
            base_score += 0.1
        
        # Check chapter outline quality
        chapter_outline = output_data.get("chapter_outline", [])
        if len(chapter_outline) == task.input_data.get("chapter_count", 8):
            base_score += 0.1
        
        # Check for detailed requirements
        research_requirements = output_data.get("research_requirements", {})
        if len(research_requirements.get("primary_sources", [])) > 0:
            base_score += 0.05
        
        # Check for timeline estimation
        if output_data.get("estimated_timeline"):
            base_score += 0.05
        
        return max(0.0, min(1.0, base_score))