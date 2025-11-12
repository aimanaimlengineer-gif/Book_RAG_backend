"""
Fact Checking Agent - Verifies accuracy and credibility of research content
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re

from agents_base_agent import BaseAgent, BaseAgentTask, BaseAgentResult, AgentCapability

logger = logging.getLogger(__name__)

class FactCheckingAgent(BaseAgent):
    """Agent responsible for fact-checking and verifying content accuracy"""
    
    def __init__(self, llm_service, web_scraping_service, config: Dict[str, Any] = None):
        self.web_scraping_service = web_scraping_service
        super().__init__(
            agent_type="fact_checking",
            role="Fact Verification Expert",
            goal="Verify accuracy, credibility, and reliability of all research content and sources",
            backstory="You are an expert fact-checker with rigorous verification methods, extensive experience in evaluating source credibility, and commitment to accuracy. You excel at cross-referencing information and identifying potential misinformation.",
            capabilities=[AgentCapability.FACT_VERIFICATION, AgentCapability.DATA_COLLECTION],
            llm_service=llm_service,
            config=config
        )
    
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """Get required capabilities for fact-checking tasks"""
        return [AgentCapability.FACT_VERIFICATION]
    
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """Execute fact-checking task"""
        try:
            input_data = task.input_data
            
            # Extract research data to verify
            research_data = input_data.get("research_data", {})
            knowledge_base = research_data.get("knowledge_base", {})
            structured_knowledge = research_data.get("structured_knowledge", {})
            category = input_data.get("category", "")
            
            # Perform comprehensive fact-checking
            verification_results = await self._verify_research_data(
                knowledge_base, structured_knowledge, category
            )
            
            # Cross-reference sources
            cross_reference_results = await self._cross_reference_sources(
                knowledge_base.get("sources", [])
            )
            
            # Verify statistical claims
            statistical_verification = await self._verify_statistical_data(
                knowledge_base.get("statistical_data", [])
            )
            
            # Check for potential bias or misinformation
            bias_analysis = await self._analyze_bias_and_misinformation(
                knowledge_base, structured_knowledge
            )
            
            # Generate credibility scores
            credibility_scores = await self._calculate_credibility_scores(
                verification_results, cross_reference_results, statistical_verification
            )
            
            # Create verification report
            verification_report = await self._generate_verification_report(
                verification_results, cross_reference_results, bias_analysis, credibility_scores
            )
            
            output_data = {
                "verified_research": {
                    "knowledge_base": self._apply_verification_filters(knowledge_base, credibility_scores),
                    "structured_knowledge": self._apply_verification_to_structured_knowledge(
                        structured_knowledge, credibility_scores
                    ),
                    "verification_metadata": {
                        "verification_date": datetime.utcnow().isoformat(),
                        "verification_method": "multi_source_cross_reference",
                        "total_sources_verified": len(knowledge_base.get("sources", [])),
                        "credibility_threshold": self.config.get("credibility_threshold", 0.7)
                    }
                },
                "verification_results": verification_results,
                "cross_reference_results": cross_reference_results,
                "statistical_verification": statistical_verification,
                "bias_analysis": bias_analysis,
                "credibility_scores": credibility_scores,
                "verification_report": verification_report,
                "quality_metrics": self._calculate_verification_quality_metrics(
                    verification_results, credibility_scores
                )
            }
            
            quality_score = self._calculate_fact_checking_quality_score(output_data, task)
            
            return BaseAgentResult(
                success=True,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=0.0,
                output_data=output_data,
                quality_score=quality_score,
                metadata={
                    "sources_verified": len(knowledge_base.get("sources", [])),
                    "high_credibility_sources": len([s for s in credibility_scores.values() if s.get("overall_score", 0) > 0.8]),
                    "flagged_sources": len([s for s in credibility_scores.values() if s.get("overall_score", 0) < 0.5])
                }
            )
            
        except Exception as e:
            logger.error(f"Fact checking failed: {str(e)}")
            raise
    
    async def _verify_research_data(
        self,
        knowledge_base: Dict[str, Any],
        structured_knowledge: Dict[str, Any],
        category: str
    ) -> Dict[str, Any]:
        """Verify research data accuracy"""
        
        verification_results = {
            "source_verification": [],
            "content_verification": [],
            "claim_verification": [],
            "overall_accuracy_score": 0.0
        }
        
        sources = knowledge_base.get("sources", [])
        
        for source in sources:
            source_verification = await self._verify_single_source(source, category)
            verification_results["source_verification"].append(source_verification)
        
        # Verify content claims
        for chapter_key, chapter_data in structured_knowledge.items():
            chapter_verification = await self._verify_chapter_content(chapter_data, category)
            verification_results["content_verification"].append({
                "chapter": chapter_key,
                **chapter_verification
            })
        
        # Calculate overall accuracy score
        if verification_results["source_verification"]:
            avg_source_score = sum(
                v.get("accuracy_score", 0) for v in verification_results["source_verification"]
            ) / len(verification_results["source_verification"])
            verification_results["overall_accuracy_score"] = avg_source_score
        
        return verification_results
    
    async def _verify_single_source(self, source: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Verify a single source"""
        
        verification_prompt = f"""
        Verify the credibility and accuracy of this source for {category} content:
        
        Title: {source.get('title', 'Unknown')}
        Author: {source.get('author', 'Unknown')}
        URL: {source.get('url', 'No URL')}
        Content Preview: {source.get('content', '')[:500]}...
        
        Evaluate:
        1. Source credibility (author expertise, publication reputation)
        2. Content accuracy (factual correctness, up-to-date information)
        3. Potential bias or conflicts of interest
        4. Relevance to {category} field
        5. Citation and reference quality
        
        Provide scores (0.0-1.0) for each criterion and overall assessment.
        Respond in JSON format.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=verification_prompt,
                max_tokens=800,
                temperature=0.2
            )
            
            verification_data = json.loads(response)
            
            # Add source identification
            verification_data["source_id"] = source.get("url", source.get("title", "unknown"))
            verification_data["verified_at"] = datetime.utcnow().isoformat()
            
            return verification_data
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Source verification failed, using fallback: {str(e)}")
            
            return {
                "source_id": source.get("url", source.get("title", "unknown")),
                "credibility_score": 0.5,
                "accuracy_score": 0.5,
                "bias_score": 0.5,
                "relevance_score": 0.6,
                "overall_score": 0.5,
                "verification_notes": "Automated verification failed, manual review recommended",
                "verified_at": datetime.utcnow().isoformat()
            }
    
    async def _verify_chapter_content(self, chapter_data: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Verify chapter content for factual accuracy"""
        
        key_concepts = chapter_data.get("key_concepts", [])
        relevant_sources = chapter_data.get("relevant_sources", [])
        
        content_verification_prompt = f"""
        Verify the factual accuracy of these key concepts for {category}:
        
        Key Concepts: {', '.join(key_concepts)}
        
        Based on current knowledge, evaluate:
        1. Factual correctness of each concept
        2. Currency of information (is it up-to-date?)
        3. Completeness of coverage
        4. Potential misconceptions or errors
        5. Need for additional verification
        
        Provide verification status for each concept and overall assessment.
        Respond in JSON format.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=content_verification_prompt,
                max_tokens=1000,
                temperature=0.2
            )
            
            content_verification = json.loads(response)
            
            # Add metadata
            content_verification["concepts_verified"] = len(key_concepts)
            content_verification["sources_referenced"] = len(relevant_sources)
            content_verification["verified_at"] = datetime.utcnow().isoformat()
            
            return content_verification
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Content verification failed: {str(e)}")
            
            return {
                "concepts_verified": len(key_concepts),
                "accuracy_score": 0.7,
                "currency_score": 0.7,
                "completeness_score": 0.6,
                "overall_verification": "partial",
                "notes": "Automated verification incomplete"
            }
    
    async def _cross_reference_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-reference sources for consistency"""
        
        cross_reference_results = {
            "consistency_analysis": [],
            "conflicting_information": [],
            "supporting_evidence": [],
            "consensus_score": 0.0
        }
        
        # Group sources by topic/claim
        source_claims = {}
        
        for source in sources:
            content = source.get("content", "")
            source_id = source.get("url", source.get("title", "unknown"))
            
            # Extract key claims (simplified)
            claims = self._extract_claims_from_content(content)
            
            for claim in claims:
                if claim not in source_claims:
                    source_claims[claim] = []
                
                source_claims[claim].append({
                    "source_id": source_id,
                    "credibility": source.get("credibility_score", 0.5)
                })
        
        # Analyze consensus for each claim
        for claim, supporting_sources in source_claims.items():
            if len(supporting_sources) > 1:
                avg_credibility = sum(s["credibility"] for s in supporting_sources) / len(supporting_sources)
                
                cross_reference_results["supporting_evidence"].append({
                    "claim": claim,
                    "source_count": len(supporting_sources),
                    "average_credibility": avg_credibility,
                    "consensus_strength": min(1.0, len(supporting_sources) / 3)  # Max consensus at 3+ sources
                })
        
        # Calculate overall consensus score
        if cross_reference_results["supporting_evidence"]:
            consensus_scores = [
                evidence["consensus_strength"] 
                for evidence in cross_reference_results["supporting_evidence"]
            ]
            cross_reference_results["consensus_score"] = sum(consensus_scores) / len(consensus_scores)
        
        return cross_reference_results
    
    async def _verify_statistical_data(self, statistical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify statistical claims and data"""
        
        statistical_verification = {
            "verified_statistics": [],
            "questionable_statistics": [],
            "missing_citations": [],
            "overall_reliability": 0.0
        }
        
        for stat_source in statistical_data:
            source_name = stat_source.get("source_name", "Unknown")
            reliability = stat_source.get("reliability", "unknown")
            
            # Verify statistical source reliability
            verification_score = self._assess_statistical_source_reliability(stat_source)
            
            if verification_score > 0.7:
                statistical_verification["verified_statistics"].append({
                    "source": source_name,
                    "reliability_score": verification_score,
                    "verification_status": "verified"
                })
            else:
                statistical_verification["questionable_statistics"].append({
                    "source": source_name,
                    "reliability_score": verification_score,
                    "issues": "Low reliability or missing verification data"
                })
        
        # Calculate overall reliability
        if statistical_data:
            all_scores = []
            for stat in statistical_verification["verified_statistics"]:
                all_scores.append(stat["reliability_score"])
            for stat in statistical_verification["questionable_statistics"]:
                all_scores.append(stat["reliability_score"])
            
            if all_scores:
                statistical_verification["overall_reliability"] = sum(all_scores) / len(all_scores)
        
        return statistical_verification
    
    async def _analyze_bias_and_misinformation(
        self,
        knowledge_base: Dict[str, Any],
        structured_knowledge: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze potential bias and misinformation"""
        
        bias_analysis_prompt = f"""
        Analyze potential bias and misinformation in this content collection:
        
        Number of sources: {len(knowledge_base.get('sources', []))}
        Content areas covered: {list(structured_knowledge.keys())}
        
        Evaluate:
        1. Source diversity (variety of perspectives and publishers)
        2. Potential commercial or political bias
        3. Representation balance
        4. Recency and currency of information
        5. Presence of controversial or disputed claims
        
        Identify any red flags or areas needing additional verification.
        Provide bias risk assessment and recommendations.
        Respond in JSON format.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=bias_analysis_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            bias_analysis = json.loads(response)
            
            # Add quantitative metrics
            sources = knowledge_base.get("sources", [])
            unique_domains = set()
            for source in sources:
                url = source.get("url", "")
                if url:
                    domain = url.split("//")[-1].split("/")[0]
                    unique_domains.add(domain)
            
            bias_analysis["source_diversity_metrics"] = {
                "total_sources": len(sources),
                "unique_domains": len(unique_domains),
                "diversity_ratio": len(unique_domains) / max(len(sources), 1)
            }
            
            return bias_analysis
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Bias analysis failed: {str(e)}")
            
            return {
                "bias_risk": "unknown",
                "diversity_score": 0.5,
                "misinformation_risk": "low",
                "recommendations": ["Manual review recommended due to analysis failure"]
            }
    
    async def _calculate_credibility_scores(
        self,
        verification_results: Dict[str, Any],
        cross_reference_results: Dict[str, Any],
        statistical_verification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive credibility scores"""
        
        credibility_scores = {}
        
        # Process source verifications
        for source_verification in verification_results.get("source_verification", []):
            source_id = source_verification.get("source_id", "unknown")
            
            credibility_scores[source_id] = {
                "accuracy_score": source_verification.get("accuracy_score", 0.5),
                "credibility_score": source_verification.get("credibility_score", 0.5),
                "bias_score": source_verification.get("bias_score", 0.5),
                "relevance_score": source_verification.get("relevance_score", 0.5),
                "cross_reference_boost": 0.0,
                "overall_score": 0.0
            }
            
            # Apply cross-reference boost
            for evidence in cross_reference_results.get("supporting_evidence", []):
                if source_id in [s["source_id"] for s in evidence.get("supporting_sources", [])]:
                    credibility_scores[source_id]["cross_reference_boost"] = evidence.get("consensus_strength", 0) * 0.1
            
            # Calculate overall score
            base_scores = [
                credibility_scores[source_id]["accuracy_score"],
                credibility_scores[source_id]["credibility_score"], 
                credibility_scores[source_id]["relevance_score"]
            ]
            
            base_average = sum(base_scores) / len(base_scores)
            boost = credibility_scores[source_id]["cross_reference_boost"]
            
            credibility_scores[source_id]["overall_score"] = min(1.0, base_average + boost)
        
        return credibility_scores
    
    async def _generate_verification_report(
        self,
        verification_results: Dict[str, Any],
        cross_reference_results: Dict[str, Any],
        bias_analysis: Dict[str, Any],
        credibility_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        
        # Calculate summary statistics
        total_sources = len(credibility_scores)
        high_credibility = len([s for s in credibility_scores.values() if s["overall_score"] > 0.8])
        medium_credibility = len([s for s in credibility_scores.values() if 0.5 <= s["overall_score"] <= 0.8])
        low_credibility = len([s for s in credibility_scores.values() if s["overall_score"] < 0.5])
        
        overall_credibility = sum(s["overall_score"] for s in credibility_scores.values()) / max(total_sources, 1)
        
        verification_report = {
            "summary": {
                "total_sources_verified": total_sources,
                "high_credibility_sources": high_credibility,
                "medium_credibility_sources": medium_credibility,
                "low_credibility_sources": low_credibility,
                "overall_credibility_score": overall_credibility
            },
            "quality_assessment": {
                "accuracy_rating": verification_results.get("overall_accuracy_score", 0.0),
                "consensus_rating": cross_reference_results.get("consensus_score", 0.0),
                "bias_risk_rating": bias_analysis.get("bias_risk", "unknown"),
                "overall_quality": "high" if overall_credibility > 0.8 else "medium" if overall_credibility > 0.6 else "needs_improvement"
            },
            "recommendations": self._generate_verification_recommendations(
                verification_results, credibility_scores, bias_analysis
            ),
            "flags_and_warnings": self._generate_verification_flags(
                credibility_scores, bias_analysis
            ),
            "next_steps": [
                "Review low-credibility sources",
                "Seek additional verification for contested claims", 
                "Consider supplementing with authoritative sources"
            ] if low_credibility > 0 else ["Content verified and ready for use"],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return verification_report
    
    def _extract_claims_from_content(self, content: str) -> List[str]:
        """Extract key claims from content (simplified)"""
        
        # Simple claim extraction based on sentence patterns
        sentences = content.split('. ')
        claims = []
        
        # Look for factual statement patterns
        claim_patterns = [
            r'\d+%', r'\d+ percent', r'according to', r'research shows', 
            r'studies indicate', r'data reveals', r'statistics show'
        ]
        
        for sentence in sentences:
            if any(re.search(pattern, sentence.lower()) for pattern in claim_patterns):
                claims.append(sentence.strip()[:100])  # Truncate long claims
        
        return claims[:10]  # Limit to top 10 claims
    
    def _assess_statistical_source_reliability(self, stat_source: Dict[str, Any]) -> float:
        """Assess reliability of statistical source"""
        
        source_name = stat_source.get("source_name", "").lower()
        reliability_indicator = stat_source.get("reliability", "unknown").lower()
        
        # Known reliable sources get higher scores
        reliable_sources = [
            "government", "federal", "bureau", "statistics", "census",
            "world bank", "imf", "oecd", "who", "cdc", "fda", "academic", "university"
        ]
        
        if any(keyword in source_name for keyword in reliable_sources):
            base_score = 0.9
        elif reliability_indicator in ["high", "verified", "official"]:
            base_score = 0.8
        elif reliability_indicator in ["medium", "established"]:
            base_score = 0.6
        else:
            base_score = 0.5
        
        return base_score
    
    def _apply_verification_filters(
        self,
        knowledge_base: Dict[str, Any],
        credibility_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply verification filters to knowledge base"""
        
        threshold = self.config.get("credibility_threshold", 0.7)
        filtered_knowledge_base = knowledge_base.copy()
        
        # Filter sources based on credibility scores
        original_sources = knowledge_base.get("sources", [])
        filtered_sources = []
        
        for source in original_sources:
            source_id = source.get("url", source.get("title", "unknown"))
            credibility_info = credibility_scores.get(source_id, {})
            overall_score = credibility_info.get("overall_score", 0.0)
            
            if overall_score >= threshold:
                # Add verification metadata to source
                source_copy = source.copy()
                source_copy["verification_metadata"] = {
                    "verified": True,
                    "credibility_score": overall_score,
                    "verification_date": datetime.utcnow().isoformat()
                }
                filtered_sources.append(source_copy)
            else:
                logger.warning(f"Source {source_id} filtered out due to low credibility: {overall_score}")
        
        filtered_knowledge_base["sources"] = filtered_sources
        filtered_knowledge_base["verification_summary"] = {
            "original_source_count": len(original_sources),
            "filtered_source_count": len(filtered_sources),
            "credibility_threshold": threshold,
            "filtered_at": datetime.utcnow().isoformat()
        }
        
        return filtered_knowledge_base
    
    def _apply_verification_to_structured_knowledge(
        self,
        structured_knowledge: Dict[str, Any],
        credibility_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply verification results to structured knowledge"""
        
        verified_structured_knowledge = {}
        
        for chapter_key, chapter_data in structured_knowledge.items():
            verified_chapter_data = chapter_data.copy()
            
            # Filter relevant sources based on credibility
            relevant_sources = chapter_data.get("relevant_sources", [])
            verified_sources = []
            
            for source in relevant_sources:
                source_id = source.get("url", source.get("title", "unknown"))
                credibility_info = credibility_scores.get(source_id, {})
                overall_score = credibility_info.get("overall_score", 0.0)
                
                if overall_score >= self.config.get("credibility_threshold", 0.7):
                    verified_sources.append(source)
            
            verified_chapter_data["relevant_sources"] = verified_sources
            verified_chapter_data["verification_metadata"] = {
                "sources_verified": len(relevant_sources),
                "sources_passed": len(verified_sources),
                "verification_rate": len(verified_sources) / max(len(relevant_sources), 1),
                "verified_at": datetime.utcnow().isoformat()
            }
            
            verified_structured_knowledge[chapter_key] = verified_chapter_data
        
        return verified_structured_knowledge
    
    def _generate_verification_recommendations(
        self,
        verification_results: Dict[str, Any],
        credibility_scores: Dict[str, Any],
        bias_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate verification recommendations"""
        
        recommendations = []
        
        # Source quality recommendations
        low_credibility_count = len([s for s in credibility_scores.values() if s["overall_score"] < 0.5])
        
        if low_credibility_count > 0:
            recommendations.append(f"Replace {low_credibility_count} low-credibility sources with authoritative alternatives")
        
        # Diversity recommendations
        diversity_metrics = bias_analysis.get("source_diversity_metrics", {})
        diversity_ratio = diversity_metrics.get("diversity_ratio", 1.0)
        
        if diversity_ratio < 0.5:
            recommendations.append("Increase source diversity to reduce potential bias")
        
        # Statistical verification recommendations
        overall_accuracy = verification_results.get("overall_accuracy_score", 0.0)
        
        if overall_accuracy < 0.8:
            recommendations.append("Seek additional verification for statistical claims")
        
        # General quality recommendations
        if not recommendations:
            recommendations.append("Content verification passed - ready for use")
        
        return recommendations
    
    def _generate_verification_flags(
        self,
        credibility_scores: Dict[str, Any],
        bias_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate verification flags and warnings"""
        
        flags = []
        
        # Check for high-risk sources
        very_low_credibility = [
            source_id for source_id, scores in credibility_scores.items()
            if scores["overall_score"] < 0.3
        ]
        
        if very_low_credibility:
            flags.append(f"HIGH RISK: {len(very_low_credibility)} sources with very low credibility")
        
        # Check bias indicators
        bias_risk = bias_analysis.get("bias_risk", "unknown")
        if bias_risk in ["high", "moderate"]:
            flags.append(f"BIAS WARNING: {bias_risk} bias risk detected")
        
        # Check source diversity
        diversity_metrics = bias_analysis.get("source_diversity_metrics", {})
        if diversity_metrics.get("diversity_ratio", 1.0) < 0.3:
            flags.append("DIVERSITY WARNING: Limited source diversity detected")
        
        return flags
    
    def _calculate_verification_quality_metrics(
        self,
        verification_results: Dict[str, Any],
        credibility_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate quality metrics for verification process"""
        
        total_sources = len(credibility_scores)
        
        if total_sources == 0:
            return {"overall_quality": 0.0, "verification_completeness": 0.0}
        
        high_quality_sources = len([s for s in credibility_scores.values() if s["overall_score"] > 0.8])
        avg_credibility = sum(s["overall_score"] for s in credibility_scores.values()) / total_sources
        
        return {
            "overall_quality": avg_credibility,
            "verification_completeness": 1.0,  # All sources processed
            "high_quality_ratio": high_quality_sources / total_sources,
            "total_sources_processed": total_sources,
            "verification_accuracy": verification_results.get("overall_accuracy_score", 0.0)
        }
    
    def _calculate_fact_checking_quality_score(
        self,
        output_data: Dict[str, Any],
        task: BaseAgentTask
    ) -> float:
        """Calculate quality score for fact-checking process"""
        
        base_score = 0.7
        
        # Quality based on verification completeness
        quality_metrics = output_data.get("quality_metrics", {})
        verification_completeness = quality_metrics.get("verification_completeness", 0.0)
        base_score += verification_completeness * 0.1
        
        # Quality based on credibility scores
        overall_quality = quality_metrics.get("overall_quality", 0.0)
        base_score += overall_quality * 0.1
        
        # Quality based on verification report
        verification_report = output_data.get("verification_report", {})
        quality_assessment = verification_report.get("quality_assessment", {})
        
        if quality_assessment.get("overall_quality") == "high":
            base_score += 0.1
        elif quality_assessment.get("overall_quality") == "medium":
            base_score += 0.05
        
        return max(0.0, min(1.0, base_score))