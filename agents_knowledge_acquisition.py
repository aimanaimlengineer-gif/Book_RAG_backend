"""
Knowledge Acquisition Agent - Gathers and curates information from multiple sources
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re

from agents_base_agent import BaseAgent, BaseAgentTask, BaseAgentResult, AgentCapability

logger = logging.getLogger(__name__)

class KnowledgeAcquisitionAgent(BaseAgent):
    """Agent responsible for gathering and curating knowledge from multiple sources"""
    
    def __init__(self, llm_service, web_scraping_service, vector_db_service, config: Dict[str, Any] = None):
        self.web_scraping_service = web_scraping_service
        self.vector_db_service = vector_db_service
        super().__init__(
            agent_type="knowledge_acquisition",
            role="Knowledge Research Specialist",
            goal="Gather, curate, and organize comprehensive information from diverse sources for book content",
            backstory="You are an expert researcher with access to vast information sources. You excel at finding, evaluating, and organizing relevant content from academic papers, industry reports, expert interviews, and authoritative web sources.",
            capabilities=[AgentCapability.DATA_COLLECTION, AgentCapability.CONTENT_EXTRACTION],
            llm_service=llm_service,
            config=config
        )
    
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """Get required capabilities for knowledge acquisition tasks"""
        return [AgentCapability.DATA_COLLECTION, AgentCapability.CONTENT_EXTRACTION]
    
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """Execute knowledge acquisition task"""
        try:
            input_data = task.input_data
            
            # Extract task parameters
            research_plan = input_data.get("research_plan", {})
            chapter_outline = input_data.get("chapter_outline", [])
            research_requirements = input_data.get("research_requirements", {})
            category = input_data.get("category", "")
            language = input_data.get("language", "en")
            
            # Execute knowledge acquisition workflow
            knowledge_base = await self._acquire_knowledge(
                research_plan, chapter_outline, research_requirements, category, language
            )
            
            # Organize and structure knowledge
            structured_knowledge = await self._structure_knowledge(knowledge_base, chapter_outline)
            
            # Create knowledge index
            knowledge_index = await self._create_knowledge_index(structured_knowledge)
            
            # Generate source citations
            citations = await self._generate_citations(knowledge_base)
            
            output_data = {
                "knowledge_base": knowledge_base,
                "structured_knowledge": structured_knowledge,
                "knowledge_index": knowledge_index,
                "citations": citations,
                "source_summary": self._create_source_summary(knowledge_base),
                "quality_metrics": self._calculate_knowledge_quality_metrics(knowledge_base)
            }
            
            quality_score = self._calculate_acquisition_quality_score(output_data, task)
            
            return BaseAgentResult(
                success=True,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=0.0,
                output_data=output_data,
                quality_score=quality_score,
                metadata={
                    "sources_processed": len(knowledge_base.get("sources", [])),
                    "total_content_length": sum(len(s.get("content", "")) for s in knowledge_base.get("sources", [])),
                    "knowledge_areas_covered": len(structured_knowledge.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Knowledge acquisition failed: {str(e)}")
            raise
    
    async def _acquire_knowledge(
        self,
        research_plan: Dict[str, Any],
        chapter_outline: List[Dict[str, Any]],
        research_requirements: Dict[str, Any],
        category: str,
        language: str
    ) -> Dict[str, Any]:
        """Execute comprehensive knowledge acquisition"""
        
        knowledge_base = {
            "sources": [],
            "primary_research": [],
            "secondary_research": [],
            "expert_insights": [],
            "statistical_data": [],
            "case_studies": [],
            "current_trends": []
        }
        
        # Gather web-based sources
        web_sources = await self._gather_web_sources(research_plan, research_requirements)
        knowledge_base["sources"].extend(web_sources)
        
        # Gather academic and professional sources
        academic_sources = await self._gather_academic_sources(research_plan, category)
        knowledge_base["secondary_research"].extend(academic_sources)
        
        # Gather statistical data
        statistical_data = await self._gather_statistical_data(research_requirements, category)
        knowledge_base["statistical_data"].extend(statistical_data)
        
        # Gather case studies
        case_studies = await self._gather_case_studies(research_requirements, category)
        knowledge_base["case_studies"].extend(case_studies)
        
        # Gather current trends and insights
        trends = await self._gather_current_trends(research_plan, category)
        knowledge_base["current_trends"].extend(trends)
        
        return knowledge_base
    
    async def _gather_web_sources(
        self,
        research_plan: Dict[str, Any],
        research_requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Gather information from web sources"""
        
        web_sources = []
        
        try:
            # Extract search terms from research plan
            search_terms = self._extract_search_terms(research_plan)
            
            # Use web scraping service if available
            if self.web_scraping_service:
                for term in search_terms[:10]:  # Limit to avoid overwhelming
                    try:
                        search_results = await self.web_scraping_service.search_and_extract(
                            query=term,
                            max_results=5,
                            source_types=["articles", "reports", "guides"]
                        )
                        
                        for result in search_results:
                            web_sources.append({
                                "source_type": "web",
                                "title": result.get("title", ""),
                                "url": result.get("url", ""),
                                "content": result.get("content", ""),
                                "author": result.get("author", "Unknown"),
                                "publish_date": result.get("publish_date"),
                                "credibility_score": result.get("credibility_score", 0.5),
                                "relevance_score": await self._calculate_relevance_score(
                                    result.get("content", ""), term
                                ),
                                "search_term": term,
                                "extracted_at": datetime.utcnow().isoformat()
                            })
                            
                    except Exception as e:
                        logger.warning(f"Failed to gather web sources for term '{term}': {str(e)}")
                        continue
            
            # Fallback: Generate simulated high-quality sources
            else:
                web_sources = await self._generate_fallback_sources(search_terms)
                
        except Exception as e:
            logger.error(f"Web source gathering failed: {str(e)}")
            
        return web_sources
    
    async def _gather_academic_sources(
        self,
        research_plan: Dict[str, Any],
        category: str
    ) -> List[Dict[str, Any]]:
        """Gather academic and professional sources"""
        
        academic_sources = []
        
        # Generate academic source references based on category
        academic_templates = {
            "technology": [
                "IEEE Computer Society publications",
                "ACM Digital Library",
                "arXiv.org computer science papers",
                "Google Scholar technical papers"
            ],
            "business": [
                "Harvard Business Review",
                "McKinsey Global Institute reports",
                "Deloitte industry insights",
                "MIT Sloan Management Review"
            ],
            "health": [
                "PubMed medical research",
                "World Health Organization reports",
                "Journal of the American Medical Association",
                "Cochrane systematic reviews"
            ],
            "finance": [
                "Federal Reserve economic data",
                "IMF working papers",
                "Financial Industry Regulatory Authority",
                "CFA Institute research"
            ]
        }
        
        template_sources = academic_templates.get(category, [
            "Peer-reviewed journal articles",
            "Industry association reports",
            "Government publications",
            "Professional organization studies"
        ])
        
        for source_type in template_sources:
            academic_sources.append({
                "source_type": "academic",
                "source_name": source_type,
                "credibility_score": 0.9,
                "access_method": "database_search",
                "content_type": "research_paper",
                "peer_reviewed": True,
                "citation_format": "APA"
            })
        
        return academic_sources
    
    async def _gather_statistical_data(
        self,
        research_requirements: Dict[str, Any],
        category: str
    ) -> List[Dict[str, Any]]:
        """Gather relevant statistical data"""
        
        statistical_data = []
        
        # Category-specific statistical sources
        stat_sources = {
            "technology": [
                "Stack Overflow Developer Survey",
                "GitHub State of the Octoverse",
                "Gartner Technology Trends",
                "IDC Market Research"
            ],
            "business": [
                "Bureau of Labor Statistics",
                "Small Business Administration data",
                "Fortune 500 analysis",
                "McKinsey Global Survey"
            ],
            "health": [
                "CDC Health Statistics",
                "WHO Global Health Observatory",
                "National Health Interview Survey",
                "Medical expenditure data"
            ],
            "finance": [
                "Federal Reserve Economic Data",
                "Bureau of Economic Analysis",
                "World Bank financial data",
                "IMF economic indicators"
            ]
        }
        
        sources = stat_sources.get(category, ["Industry surveys", "Government statistics"])
        
        for source in sources:
            statistical_data.append({
                "source_type": "statistical",
                "source_name": source,
                "data_type": "quantitative",
                "reliability": "high",
                "update_frequency": "annual",
                "coverage": "comprehensive"
            })
        
        return statistical_data
    
    async def _gather_case_studies(
        self,
        research_requirements: Dict[str, Any],
        category: str
    ) -> List[Dict[str, Any]]:
        """Gather relevant case studies"""
        
        case_studies = []
        
        # Generate case study templates based on requirements
        case_study_requirements = research_requirements.get("case_studies", [])
        
        for requirement in case_study_requirements:
            case_studies.append({
                "case_study_type": requirement.get("case_study_type", "general"),
                "chapter": requirement.get("chapter", 1),
                "priority": requirement.get("priority", "medium"),
                "content_type": "real_world_example",
                "analysis_depth": "detailed",
                "learning_outcome": "practical_application"
            })
        
        # Add category-specific case studies
        if not case_studies:
            default_cases = {
                "technology": ["Successful software implementation", "Digital transformation", "Tech startup growth"],
                "business": ["Company turnaround", "Market expansion", "Strategic partnership"],
                "health": ["Public health intervention", "Treatment protocol success", "Wellness program"],
                "finance": ["Investment strategy", "Risk management", "Financial planning success"]
            }
            
            for case_type in default_cases.get(category, ["Industry example"]):
                case_studies.append({
                    "case_study_type": case_type,
                    "content_type": "real_world_example",
                    "analysis_depth": "detailed"
                })
        
        return case_studies
    
    async def _gather_current_trends(
        self,
        research_plan: Dict[str, Any],
        category: str
    ) -> List[Dict[str, Any]]:
        """Gather current trends and emerging insights"""
        
        trends = []
        
        # Use LLM to identify current trends
        trends_prompt = f"""
        Identify the top 5 current trends in {category} that would be relevant for a comprehensive book.
        For each trend, provide:
        1. Trend name
        2. Brief description
        3. Impact level (high/medium/low)
        4. Timeline (emerging/current/established)
        5. Relevance to general audience
        
        Focus on trends that have practical implications and educational value.
        Provide response in JSON format.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=trends_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            trends_data = json.loads(response)
            
            for trend in trends_data.get("trends", []):
                trends.append({
                    "trend_name": trend.get("name", ""),
                    "description": trend.get("description", ""),
                    "impact_level": trend.get("impact_level", "medium"),
                    "timeline": trend.get("timeline", "current"),
                    "relevance": trend.get("relevance", "medium"),
                    "source_type": "trend_analysis",
                    "identified_at": datetime.utcnow().isoformat()
                })
                
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Trend analysis failed, using fallback: {str(e)}")
            
            # Fallback trend categories
            fallback_trends = {
                "technology": ["AI/ML adoption", "Cloud computing", "Cybersecurity", "Remote work tools"],
                "business": ["Digital transformation", "Sustainability", "Remote work", "Data-driven decisions"],
                "health": ["Telemedicine", "Mental health awareness", "Preventive care", "Personalized medicine"],
                "finance": ["Digital payments", "Cryptocurrency", "Robo-advisors", "ESG investing"]
            }
            
            for trend_name in fallback_trends.get(category, ["Industry evolution"]):
                trends.append({
                    "trend_name": trend_name,
                    "description": f"Current trend in {category}",
                    "impact_level": "medium",
                    "timeline": "current",
                    "source_type": "trend_analysis"
                })
        
        return trends
    
    async def _structure_knowledge(
        self,
        knowledge_base: Dict[str, Any],
        chapter_outline: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Structure knowledge according to chapter outline"""
        
        structured_knowledge = {}
        
        for chapter in chapter_outline:
            chapter_num = chapter.get("chapter_number", 1)
            chapter_title = chapter.get("title", f"Chapter {chapter_num}")
            key_concepts = chapter.get("key_concepts", [])
            
            chapter_knowledge = {
                "title": chapter_title,
                "key_concepts": key_concepts,
                "relevant_sources": [],
                "statistical_data": [],
                "case_studies": [],
                "current_trends": [],
                "expert_insights": []
            }
            
            # Map sources to chapters based on relevance
            for source in knowledge_base.get("sources", []):
                relevance_score = await self._calculate_chapter_relevance(
                    source.get("content", ""), key_concepts
                )
                
                if relevance_score > 0.3:  # Threshold for relevance
                    chapter_knowledge["relevant_sources"].append({
                        **source,
                        "relevance_score": relevance_score
                    })
            
            # Map statistical data
            for stat_data in knowledge_base.get("statistical_data", []):
                chapter_knowledge["statistical_data"].append(stat_data)
            
            # Map case studies
            for case_study in knowledge_base.get("case_studies", []):
                if case_study.get("chapter") == chapter_num:
                    chapter_knowledge["case_studies"].append(case_study)
            
            # Map trends
            chapter_knowledge["current_trends"] = knowledge_base.get("current_trends", [])
            
            structured_knowledge[f"chapter_{chapter_num}"] = chapter_knowledge
        
        return structured_knowledge
    
    async def _create_knowledge_index(self, structured_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Create searchable knowledge index"""
        
        knowledge_index = {
            "concepts": {},
            "sources": {},
            "keywords": {},
            "cross_references": []
        }
        
        # Index concepts
        for chapter_key, chapter_data in structured_knowledge.items():
            for concept in chapter_data.get("key_concepts", []):
                if concept not in knowledge_index["concepts"]:
                    knowledge_index["concepts"][concept] = []
                
                knowledge_index["concepts"][concept].append({
                    "chapter": chapter_key,
                    "context": chapter_data.get("title", "")
                })
        
        # Index sources
        for chapter_key, chapter_data in structured_knowledge.items():
            for source in chapter_data.get("relevant_sources", []):
                source_id = source.get("url", source.get("title", "unknown"))
                knowledge_index["sources"][source_id] = {
                    "title": source.get("title", ""),
                    "chapters": [chapter_key],
                    "credibility": source.get("credibility_score", 0.5),
                    "relevance": source.get("relevance_score", 0.5)
                }
        
        return knowledge_index
    
    async def _generate_citations(self, knowledge_base: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate properly formatted citations"""
        
        citations = []
        
        for source in knowledge_base.get("sources", []):
            citation = {
                "id": len(citations) + 1,
                "type": source.get("source_type", "web"),
                "title": source.get("title", ""),
                "author": source.get("author", "Unknown"),
                "url": source.get("url", ""),
                "publish_date": source.get("publish_date"),
                "accessed_date": source.get("extracted_at", datetime.utcnow().isoformat()),
                "format": "APA"  # Default format
            }
            
            # Generate APA format citation
            if citation["type"] == "web":
                apa_citation = f"{citation['author']} ({citation['publish_date'] or 'n.d.'}). {citation['title']}. Retrieved from {citation['url']}"
            else:
                apa_citation = f"{citation['author']} ({citation['publish_date'] or 'n.d.'}). {citation['title']}."
            
            citation["formatted_citation"] = apa_citation
            citations.append(citation)
        
        return citations
    
    def _extract_search_terms(self, research_plan: Dict[str, Any]) -> List[str]:
        """Extract search terms from research plan"""
        
        search_terms = []
        
        # Extract from core themes
        core_themes = research_plan.get("core_themes", [])
        search_terms.extend(core_themes)
        
        # Extract from key concepts
        key_concepts = research_plan.get("key_concepts", [])
        search_terms.extend(key_concepts)
        
        # Clean and deduplicate
        search_terms = list(set([term.strip() for term in search_terms if term.strip()]))
        
        return search_terms[:20]  # Limit to avoid overwhelming
    
    async def _calculate_relevance_score(self, content: str, search_term: str) -> float:
        """Calculate relevance score for content"""
        
        if not content or not search_term:
            return 0.0
        
        content_lower = content.lower()
        term_lower = search_term.lower()
        
        # Simple keyword matching
        exact_matches = content_lower.count(term_lower)
        word_matches = sum(1 for word in term_lower.split() if word in content_lower)
        
        # Calculate score based on matches and content length
        content_length = len(content.split())
        
        if content_length == 0:
            return 0.0
        
        relevance_score = (exact_matches * 2 + word_matches) / content_length * 100
        
        return min(1.0, relevance_score)
    
    async def _calculate_chapter_relevance(self, content: str, key_concepts: List[str]) -> float:
        """Calculate relevance of content to chapter concepts"""
        
        if not content or not key_concepts:
            return 0.0
        
        total_relevance = 0.0
        
        for concept in key_concepts:
            concept_relevance = await self._calculate_relevance_score(content, concept)
            total_relevance += concept_relevance
        
        return total_relevance / len(key_concepts) if key_concepts else 0.0
    
    async def _generate_fallback_sources(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """Generate fallback sources when web scraping is unavailable"""
        
        fallback_sources = []
        
        for term in search_terms[:5]:  # Limit fallback sources
            fallback_sources.append({
                "source_type": "web",
                "title": f"Comprehensive Guide to {term}",
                "url": f"https://example.com/{term.replace(' ', '-').lower()}",
                "content": f"This is a comprehensive resource about {term} covering fundamental concepts, best practices, and practical applications.",
                "author": "Industry Expert",
                "publish_date": "2024",
                "credibility_score": 0.7,
                "relevance_score": 0.8,
                "search_term": term,
                "extracted_at": datetime.utcnow().isoformat(),
                "note": "Fallback source - actual content would be gathered from web scraping"
            })
        
        return fallback_sources
    
    def _create_source_summary(self, knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of sources gathered"""
        
        return {
            "total_sources": len(knowledge_base.get("sources", [])),
            "source_types": {
                "web": len([s for s in knowledge_base.get("sources", []) if s.get("source_type") == "web"]),
                "academic": len(knowledge_base.get("secondary_research", [])),
                "statistical": len(knowledge_base.get("statistical_data", [])),
                "case_studies": len(knowledge_base.get("case_studies", [])),
                "trends": len(knowledge_base.get("current_trends", []))
            },
            "average_credibility": sum(s.get("credibility_score", 0) for s in knowledge_base.get("sources", [])) / max(len(knowledge_base.get("sources", [])), 1),
            "coverage_assessment": "comprehensive" if len(knowledge_base.get("sources", [])) > 10 else "moderate"
        }
    
    def _calculate_knowledge_quality_metrics(self, knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for gathered knowledge"""
        
        sources = knowledge_base.get("sources", [])
        
        if not sources:
            return {"overall_quality": 0.0, "completeness": 0.0, "credibility": 0.0}
        
        # Calculate average credibility
        avg_credibility = sum(s.get("credibility_score", 0) for s in sources) / len(sources)
        
        # Calculate completeness based on source diversity
        source_types = set(s.get("source_type", "unknown") for s in sources)
        completeness = min(1.0, len(source_types) / 4)  # Assuming 4 main source types
        
        # Calculate overall quality
        overall_quality = (avg_credibility + completeness) / 2
        
        return {
            "overall_quality": overall_quality,
            "completeness": completeness,
            "credibility": avg_credibility,
            "source_diversity": len(source_types),
            "total_sources": len(sources)
        }
    
    def _calculate_acquisition_quality_score(
        self,
        output_data: Dict[str, Any],
        task: BaseAgentTask
    ) -> float:
        """Calculate quality score for knowledge acquisition"""
        
        base_score = 0.6
        
        # Quality based on number of sources
        knowledge_base = output_data.get("knowledge_base", {})
        source_count = len(knowledge_base.get("sources", []))
        
        if source_count >= 20:
            base_score += 0.2
        elif source_count >= 10:
            base_score += 0.1
        
        # Quality based on source diversity
        quality_metrics = output_data.get("quality_metrics", {})
        source_diversity = quality_metrics.get("source_diversity", 0)
        
        if source_diversity >= 3:
            base_score += 0.1
        
        # Quality based on credibility
        avg_credibility = quality_metrics.get("credibility", 0)
        base_score += avg_credibility * 0.1
        
        return max(0.0, min(1.0, base_score))