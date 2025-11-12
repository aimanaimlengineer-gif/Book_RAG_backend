"""
Web Research Sub-Agent - Focused web content retrieval and analysis
Parent Agent: Knowledge Acquisition Agent
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re
from urllib.parse import urljoin, urlparse

from agents.base_agent import BaseAgent, BaseAgentTask, BaseAgentResult, AgentCapability

logger = logging.getLogger(__name__)

class WebResearchSubAgent(BaseAgent):
    """Sub-agent specialized in web content retrieval and analysis"""
    
    def __init__(self, llm_service, web_scraping_service, config: Dict[str, Any] = None):
        self.web_scraping_service = web_scraping_service
        super().__init__(
            agent_type="web_research_sub_agent",
            role="Web Research Specialist",
            goal="Retrieve and analyze web content for comprehensive research coverage",
            backstory="You are an expert web researcher with deep knowledge of online sources, content analysis, and trend identification. You excel at finding authoritative web content and extracting relevant information.",
            capabilities=[AgentCapability.DATA_COLLECTION, AgentCapability.CONTENT_EXTRACTION],
            llm_service=llm_service,
            config=config
        )
    
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """Get required capabilities for web research tasks"""
        return [AgentCapability.DATA_COLLECTION, AgentCapability.CONTENT_EXTRACTION]
    
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """Execute web research task"""
        try:
            input_data = task.input_data
            
            # Extract research parameters
            research_topics = input_data.get("research_topics", [])
            search_queries = input_data.get("search_queries", [])
            target_domains = input_data.get("target_domains", [])
            content_filters = input_data.get("content_filters", {})
            
            # Perform comprehensive web research
            web_sources = await self._search_web_sources(
                research_topics, search_queries, target_domains
            )
            
            # Analyze content quality and relevance
            content_analysis = await self._analyze_content_quality(
                web_sources, content_filters
            )
            
            # Extract key information
            extracted_info = await self._extract_key_information(
                web_sources, research_topics
            )
            
            # Identify trends and patterns
            trend_analysis = await self._analyze_trends_and_patterns(
                web_sources, research_topics
            )
            
            # Generate research summary
            research_summary = await self._generate_research_summary(
                web_sources, content_analysis, extracted_info, trend_analysis
            )
            
            output_data = {
                "web_sources": web_sources,
                "content_analysis": content_analysis,
                "extracted_info": extracted_info,
                "trend_analysis": trend_analysis,
                "research_summary": research_summary,
                "metadata": {
                    "sources_analyzed": len(web_sources),
                    "research_date": datetime.utcnow().isoformat(),
                    "coverage_score": self._calculate_coverage_score(web_sources, research_topics)
                }
            }
            
            quality_score = self._calculate_web_research_quality_score(output_data, task)
            
            return BaseAgentResult(
                success=True,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=0.0,
                output_data=output_data,
                quality_score=quality_score,
                metadata={
                    "sources_found": len(web_sources),
                    "high_quality_sources": len([s for s in web_sources if s.get("quality_score", 0) > 0.8]),
                    "coverage_percentage": output_data["metadata"]["coverage_score"]
                }
            )
            
        except Exception as e:
            logger.error(f"Web research failed: {str(e)}")
            raise
    
    async def _search_web_sources(
        self,
        research_topics: List[str],
        search_queries: List[str],
        target_domains: List[str]
    ) -> List[Dict[str, Any]]:
        """Search for web sources based on research topics"""
        
        web_sources = []
        
        # Combine topics and queries for comprehensive search
        all_search_terms = research_topics + search_queries
        
        for search_term in all_search_terms:
            try:
                # Use web scraping service to search
                search_results = await self.web_scraping_service.search_content(
                    query=search_term,
                    max_results=10,
                    domain_filter=target_domains
                )
                
                for result in search_results:
                    # Enhance result with additional analysis
                    enhanced_result = await self._enhance_search_result(result, search_term)
                    web_sources.append(enhanced_result)
                    
            except Exception as e:
                logger.warning(f"Search failed for term '{search_term}': {str(e)}")
                continue
        
        # Remove duplicates and sort by relevance
        unique_sources = self._deduplicate_sources(web_sources)
        return sorted(unique_sources, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    async def _enhance_search_result(
        self,
        result: Dict[str, Any],
        search_term: str
    ) -> Dict[str, Any]:
        """Enhance search result with additional analysis"""
        
        # Calculate relevance score
        relevance_score = await self._calculate_relevance_score(result, search_term)
        
        # Extract key concepts
        key_concepts = await self._extract_key_concepts(result.get("content", ""))
        
        # Determine content type
        content_type = self._classify_content_type(result)
        
        # Assess credibility
        credibility_score = await self._assess_source_credibility(result)
        
        enhanced_result = result.copy()
        enhanced_result.update({
            "relevance_score": relevance_score,
            "key_concepts": key_concepts,
            "content_type": content_type,
            "credibility_score": credibility_score,
            "search_term": search_term,
            "analyzed_at": datetime.utcnow().isoformat()
        })
        
        return enhanced_result
    
    async def _calculate_relevance_score(
        self,
        result: Dict[str, Any],
        search_term: str
    ) -> float:
        """Calculate relevance score for search result"""
        
        content = result.get("content", "").lower()
        title = result.get("title", "").lower()
        search_lower = search_term.lower()
        
        # Base relevance calculation
        relevance = 0.0
        
        # Title relevance (higher weight)
        if search_lower in title:
            relevance += 0.4
        
        # Content relevance
        content_matches = content.count(search_lower)
        if content_matches > 0:
            relevance += min(0.3, content_matches * 0.05)
        
        # URL relevance
        url = result.get("url", "").lower()
        if any(word in url for word in search_lower.split()):
            relevance += 0.1
        
        # Domain authority bonus
        domain = urlparse(result.get("url", "")).netloc
        if self._is_authoritative_domain(domain):
            relevance += 0.2
        
        return min(1.0, relevance)
    
    async def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        
        if not content:
            return []
        
        # Simple concept extraction (can be enhanced with NLP)
        concepts = []
        
        # Extract capitalized terms (potential proper nouns)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        concepts.extend(capitalized_terms[:10])  # Limit to top 10
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', content)
        concepts.extend(quoted_terms[:5])
        
        # Remove duplicates and return
        return list(set(concepts))[:15]
    
    def _classify_content_type(self, result: Dict[str, Any]) -> str:
        """Classify the type of content"""
        
        url = result.get("url", "").lower()
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()
        
        # Classification based on URL patterns and content
        if any(pattern in url for pattern in ['.edu', 'university', 'academic']):
            return "academic"
        elif any(pattern in url for pattern in ['.gov', 'government']):
            return "government"
        elif any(pattern in url for pattern in ['.org', 'organization']):
            return "organization"
        elif any(pattern in url for pattern in ['news', 'article', 'blog']):
            return "news_article"
        elif any(pattern in url for pattern in ['wiki', 'encyclopedia']):
            return "reference"
        else:
            return "general"
    
    async def _assess_source_credibility(self, result: Dict[str, Any]) -> float:
        """Assess the credibility of a source"""
        
        url = result.get("url", "")
        domain = urlparse(url).netloc.lower()
        
        # Known credible domains
        credible_domains = [
            'wikipedia.org', 'edu', 'gov', 'who.int', 'cdc.gov',
            'nih.gov', 'nature.com', 'science.org', 'reuters.com',
            'bbc.com', 'nytimes.com', 'theguardian.com'
        ]
        
        # Check domain credibility
        for credible_domain in credible_domains:
            if credible_domain in domain:
                return 0.9
        
        # Check for HTTPS
        if url.startswith('https://'):
            credibility = 0.6
        else:
            credibility = 0.4
        
        # Check content length (longer content often more credible)
        content_length = len(result.get("content", ""))
        if content_length > 1000:
            credibility += 0.1
        elif content_length < 200:
            credibility -= 0.1
        
        return max(0.0, min(1.0, credibility))
    
    def _is_authoritative_domain(self, domain: str) -> bool:
        """Check if domain is considered authoritative"""
        
        authoritative_patterns = [
            'edu', 'gov', 'org', 'wikipedia.org', 'who.int',
            'cdc.gov', 'nih.gov', 'nature.com', 'science.org'
        ]
        
        return any(pattern in domain for pattern in authoritative_patterns)
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources based on URL"""
        
        seen_urls = set()
        unique_sources = []
        
        for source in sources:
            url = source.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        return unique_sources
    
    async def _analyze_content_quality(
        self,
        web_sources: List[Dict[str, Any]],
        content_filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the quality of web content"""
        
        quality_analysis = {
            "overall_quality_score": 0.0,
            "source_diversity": 0.0,
            "content_freshness": 0.0,
            "authority_distribution": {},
            "quality_issues": []
        }
        
        if not web_sources:
            return quality_analysis
        
        # Calculate overall quality score
        quality_scores = [s.get("credibility_score", 0.5) for s in web_sources]
        quality_analysis["overall_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        # Analyze source diversity
        domains = [urlparse(s.get("url", "")).netloc for s in web_sources]
        unique_domains = set(domains)
        quality_analysis["source_diversity"] = len(unique_domains) / len(domains)
        
        # Analyze content freshness
        current_date = datetime.utcnow()
        fresh_sources = 0
        
        for source in web_sources:
            # Simple freshness check (can be enhanced with actual date parsing)
            if "2024" in source.get("content", "") or "2023" in source.get("content", ""):
                fresh_sources += 1
        
        quality_analysis["content_freshness"] = fresh_sources / len(web_sources)
        
        # Analyze authority distribution
        for source in web_sources:
            content_type = source.get("content_type", "general")
            quality_analysis["authority_distribution"][content_type] = \
                quality_analysis["authority_distribution"].get(content_type, 0) + 1
        
        return quality_analysis
    
    async def _extract_key_information(
        self,
        web_sources: List[Dict[str, Any]],
        research_topics: List[str]
    ) -> Dict[str, Any]:
        """Extract key information from web sources"""
        
        key_info = {
            "factual_claims": [],
            "statistics": [],
            "expert_opinions": [],
            "definitions": [],
            "examples": []
        }
        
        for source in web_sources:
            content = source.get("content", "")
            
            # Extract factual claims
            factual_claims = self._extract_factual_claims(content)
            key_info["factual_claims"].extend(factual_claims)
            
            # Extract statistics
            statistics = self._extract_statistics(content)
            key_info["statistics"].extend(statistics)
            
            # Extract definitions
            definitions = self._extract_definitions(content)
            key_info["definitions"].extend(definitions)
        
        # Limit results to prevent overwhelming output
        for key in key_info:
            key_info[key] = key_info[key][:20]
        
        return key_info
    
    def _extract_factual_claims(self, content: str) -> List[str]:
        """Extract factual claims from content"""
        
        # Simple pattern matching for factual claims
        patterns = [
            r'[A-Z][^.]*is\s+[^.]*\.',
            r'[A-Z][^.]*are\s+[^.]*\.',
            r'[A-Z][^.]*has\s+[^.]*\.',
            r'[A-Z][^.]*have\s+[^.]*\.'
        ]
        
        claims = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            claims.extend(matches)
        
        return claims[:10]  # Limit to top 10
    
    def _extract_statistics(self, content: str) -> List[str]:
        """Extract statistical information from content"""
        
        # Pattern for statistics
        stat_pattern = r'\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s+(?:percent|million|billion|thousand)'
        statistics = re.findall(stat_pattern, content, re.IGNORECASE)
        
        return statistics[:10]  # Limit to top 10
    
    def _extract_definitions(self, content: str) -> List[str]:
        """Extract definitions from content"""
        
        # Pattern for definitions
        def_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|refers to|means)\s+([^.]*\.)'
        definitions = re.findall(def_pattern, content)
        
        return [f"{term}: {definition}" for term, definition in definitions[:10]]
    
    async def _analyze_trends_and_patterns(
        self,
        web_sources: List[Dict[str, Any]],
        research_topics: List[str]
    ) -> Dict[str, Any]:
        """Analyze trends and patterns in web content"""
        
        trend_analysis = {
            "common_themes": [],
            "emerging_topics": [],
            "controversial_points": [],
            "consensus_areas": []
        }
        
        # Analyze common themes
        all_concepts = []
        for source in web_sources:
            all_concepts.extend(source.get("key_concepts", []))
        
        # Count concept frequency
        concept_counts = {}
        for concept in all_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Get most common themes
        common_themes = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        trend_analysis["common_themes"] = [theme for theme, count in common_themes[:10]]
        
        return trend_analysis
    
    async def _generate_research_summary(
        self,
        web_sources: List[Dict[str, Any]],
        content_analysis: Dict[str, Any],
        extracted_info: Dict[str, Any],
        trend_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive research summary"""
        
        summary = {
            "total_sources": len(web_sources),
            "quality_assessment": content_analysis["overall_quality_score"],
            "key_findings": [],
            "research_gaps": [],
            "recommendations": []
        }
        
        # Generate key findings
        if extracted_info["factual_claims"]:
            summary["key_findings"].append(f"Found {len(extracted_info['factual_claims'])} factual claims")
        
        if extracted_info["statistics"]:
            summary["key_findings"].append(f"Identified {len(extracted_info['statistics'])} statistical data points")
        
        # Identify research gaps
        if content_analysis["source_diversity"] < 0.5:
            summary["research_gaps"].append("Limited source diversity - need more varied sources")
        
        if content_analysis["content_freshness"] < 0.3:
            summary["research_gaps"].append("Outdated content - need more recent sources")
        
        # Generate recommendations
        if content_analysis["overall_quality_score"] < 0.7:
            summary["recommendations"].append("Seek higher quality, more authoritative sources")
        
        if not trend_analysis["common_themes"]:
            summary["recommendations"].append("Expand research scope to identify key themes")
        
        return summary
    
    def _calculate_coverage_score(
        self,
        web_sources: List[Dict[str, Any]],
        research_topics: List[str]
    ) -> float:
        """Calculate research coverage score"""
        
        if not research_topics or not web_sources:
            return 0.0
        
        covered_topics = 0
        for topic in research_topics:
            topic_lower = topic.lower()
            for source in web_sources:
                content = source.get("content", "").lower()
                if topic_lower in content:
                    covered_topics += 1
                    break
        
        return covered_topics / len(research_topics)
    
    def _calculate_web_research_quality_score(
        self,
        output_data: Dict[str, Any],
        task: BaseAgentTask
    ) -> float:
        """Calculate quality score for web research"""
        
        base_score = 0.7
        
        # Quality based on source count
        source_count = output_data["metadata"]["sources_analyzed"]
        if source_count >= 20:
            base_score += 0.1
        elif source_count >= 10:
            base_score += 0.05
        
        # Quality based on coverage
        coverage_score = output_data["metadata"]["coverage_score"]
        base_score += coverage_score * 0.1
        
        # Quality based on content analysis
        content_analysis = output_data.get("content_analysis", {})
        overall_quality = content_analysis.get("overall_quality_score", 0.5)
        base_score += overall_quality * 0.1
        
        return max(0.0, min(1.0, base_score))
