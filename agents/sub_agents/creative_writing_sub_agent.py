"""
Creative Writing Sub-Agent - Generates creative and engaging prose
Parent Agent: Content Generation Agent
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re

from agents.base_agent import BaseAgent, BaseAgentTask, BaseAgentResult, AgentCapability

logger = logging.getLogger(__name__)

class CreativeWritingSubAgent(BaseAgent):
    """Sub-agent specialized in creative writing and storytelling"""
    
    def __init__(self, llm_service, config: Dict[str, Any] = None):
        super().__init__(
            agent_type="creative_writing_sub_agent",
            role="Creative Writing Specialist",
            goal="Generate engaging, creative prose that captivates readers",
            backstory="You are a master storyteller and creative writer with expertise in narrative techniques, character development, and engaging prose. You excel at making complex topics accessible and entertaining.",
            capabilities=[AgentCapability.CREATIVE_WRITING, AgentCapability.CONTENT_GENERATION],
            llm_service=llm_service,
            config=config
        )
    
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """Get required capabilities for creative writing tasks"""
        return [AgentCapability.CREATIVE_WRITING]
    
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """Execute creative writing task"""
        try:
            input_data = task.input_data
            
            # Extract writing parameters
            content_outline = input_data.get("content_outline", {})
            writing_style = input_data.get("writing_style", "informative")
            target_audience = input_data.get("target_audience", "general")
            creative_requirements = input_data.get("creative_requirements", {})
            
            # Generate creative content
            creative_content = await self._generate_creative_content(
                content_outline, writing_style, target_audience, creative_requirements
            )
            
            # Apply storytelling techniques
            enhanced_content = await self._apply_storytelling_techniques(
                creative_content, writing_style, target_audience
            )
            
            # Add narrative elements
            narrative_content = await self._add_narrative_elements(
                enhanced_content, content_outline, target_audience
            )
            
            # Optimize for engagement
            engaging_content = await self._optimize_for_engagement(
                narrative_content, target_audience, creative_requirements
            )
            
            output_data = {
                "creative_content": engaging_content,
                "storytelling_elements": await self._analyze_storytelling_elements(engaging_content),
                "engagement_metrics": await self._calculate_engagement_metrics(engaging_content),
                "creative_techniques_used": await self._identify_creative_techniques(engaging_content),
                "readability_enhancements": await self._assess_readability_enhancements(engaging_content),
                "metadata": {
                    "writing_style": writing_style,
                    "target_audience": target_audience,
                    "creative_score": await self._calculate_creative_score(engaging_content),
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
            quality_score = self._calculate_creative_writing_quality_score(output_data, task)
            
            return BaseAgentResult(
                success=True,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=0.0,
                output_data=output_data,
                quality_score=quality_score,
                metadata={
                    "content_length": len(engaging_content.get("content", "")),
                    "creative_score": output_data["metadata"]["creative_score"],
                    "engagement_level": output_data["engagement_metrics"].get("overall_engagement", 0.0)
                }
            )
            
        except Exception as e:
            logger.error(f"Creative writing failed: {str(e)}")
            raise
    
    async def _generate_creative_content(
        self,
        content_outline: Dict[str, Any],
        writing_style: str,
        target_audience: str,
        creative_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate creative content based on outline"""
        
        creative_prompt = f"""
        Transform this content outline into engaging, creative prose for {target_audience}:
        
        Writing Style: {writing_style}
        Target Audience: {target_audience}
        Content Outline: {json.dumps(content_outline, indent=2)}
        Creative Requirements: {json.dumps(creative_requirements, indent=2)}
        
        Requirements:
        1. Use engaging storytelling techniques
        2. Make complex topics accessible and interesting
        3. Include vivid descriptions and examples
        4. Maintain appropriate tone for {target_audience}
        5. Create compelling narrative flow
        6. Use creative metaphors and analogies
        
        Generate creative, engaging content that brings the outline to life.
        """
        
        try:
            creative_content = await self.llm_service.generate_content(
                content="",
                prompt=creative_prompt,
                max_tokens=2000,
                temperature=0.7  # Higher temperature for creativity
            )
            
            return {
                "content": creative_content,
                "style": writing_style,
                "audience": target_audience,
                "creative_elements": self._extract_creative_elements(creative_content)
            }
            
        except Exception as e:
            logger.warning(f"Creative content generation failed: {str(e)}")
            return {
                "content": "Creative content generation failed. Please try again.",
                "style": writing_style,
                "audience": target_audience,
                "creative_elements": []
            }
    
    async def _apply_storytelling_techniques(
        self,
        content: Dict[str, Any],
        writing_style: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """Apply storytelling techniques to enhance content"""
        
        storytelling_prompt = f"""
        Enhance this content with storytelling techniques for {target_audience}:
        
        Content: {content.get('content', '')}
        Writing Style: {writing_style}
        
        Apply these storytelling techniques:
        1. Create compelling opening hooks
        2. Use the hero's journey structure where appropriate
        3. Include conflict and resolution
        4. Add character development elements
        5. Use dialogue and conversations
        6. Create emotional connections
        7. Build suspense and anticipation
        8. Use vivid sensory details
        
        Maintain the original meaning while making it more engaging and story-like.
        """
        
        try:
            enhanced_content = await self.llm_service.generate_content(
                content="",
                prompt=storytelling_prompt,
                max_tokens=len(content.get('content', '').split()) * 2,
                temperature=0.6
            )
            
            enhanced_content_dict = content.copy()
            enhanced_content_dict["content"] = enhanced_content
            enhanced_content_dict["storytelling_applied"] = True
            
            return enhanced_content_dict
            
        except Exception as e:
            logger.warning(f"Storytelling enhancement failed: {str(e)}")
            return content
    
    async def _add_narrative_elements(
        self,
        content: Dict[str, Any],
        content_outline: Dict[str, Any],
        target_audience: str
    ) -> Dict[str, Any]:
        """Add narrative elements to enhance storytelling"""
        
        narrative_prompt = f"""
        Add compelling narrative elements to this content for {target_audience}:
        
        Content: {content.get('content', '')}
        Outline: {json.dumps(content_outline, indent=2)}
        
        Add these narrative elements:
        1. Compelling characters or personas
        2. Engaging scenarios and situations
        3. Real-world examples and case studies
        4. Personal anecdotes and stories
        5. Problem-solution narratives
        6. Journey and transformation stories
        7. Before-and-after scenarios
        8. Success stories and testimonials
        
        Make the content more relatable and engaging through storytelling.
        """
        
        try:
            narrative_content = await self.llm_service.generate_content(
                content="",
                prompt=narrative_prompt,
                max_tokens=len(content.get('content', '').split()) * 2,
                temperature=0.5
            )
            
            narrative_content_dict = content.copy()
            narrative_content_dict["content"] = narrative_content
            narrative_content_dict["narrative_elements_added"] = True
            
            return narrative_content_dict
            
        except Exception as e:
            logger.warning(f"Narrative enhancement failed: {str(e)}")
            return content
    
    async def _optimize_for_engagement(
        self,
        content: Dict[str, Any],
        target_audience: str,
        creative_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize content for maximum engagement"""
        
        engagement_prompt = f"""
        Optimize this content for maximum engagement with {target_audience}:
        
        Content: {content.get('content', '')}
        Creative Requirements: {json.dumps(creative_requirements, indent=2)}
        
        Optimization techniques:
        1. Use power words and emotional triggers
        2. Create curiosity gaps and cliffhangers
        3. Add interactive elements and questions
        4. Use varied sentence structures
        5. Include compelling statistics and facts
        6. Create visual imagery with words
        7. Use rhetorical questions
        8. Add call-to-action elements
        9. Create urgency and importance
        10. Use social proof and authority
        
        Make the content irresistible to read and highly engaging.
        """
        
        try:
            engaging_content = await self.llm_service.generate_content(
                content="",
                prompt=engagement_prompt,
                max_tokens=len(content.get('content', '').split()) * 2,
                temperature=0.6
            )
            
            engaging_content_dict = content.copy()
            engaging_content_dict["content"] = engaging_content
            engaging_content_dict["engagement_optimized"] = True
            
            return engaging_content_dict
            
        except Exception as e:
            logger.warning(f"Engagement optimization failed: {str(e)}")
            return content
    
    def _extract_creative_elements(self, content: str) -> List[str]:
        """Extract creative elements from content"""
        
        creative_elements = []
        
        # Check for metaphors
        metaphor_patterns = [
            r'\b(like|as|similar to|resembles)\b.*\b(like|as|similar to|resembles)\b',
            r'\b(is|are|was|were)\b.*\b(like|as|similar to)\b'
        ]
        
        for pattern in metaphor_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                creative_elements.append("metaphors")
                break
        
        # Check for alliteration
        alliteration_pattern = r'\b(\w)\w*\s+\1\w*'
        if re.search(alliteration_pattern, content, re.IGNORECASE):
            creative_elements.append("alliteration")
        
        # Check for rhetorical questions
        if '?' in content and len(re.findall(r'\?', content)) > 2:
            creative_elements.append("rhetorical_questions")
        
        # Check for vivid descriptions
        descriptive_words = ['vivid', 'brilliant', 'stunning', 'magnificent', 'spectacular']
        if any(word in content.lower() for word in descriptive_words):
            creative_elements.append("vivid_descriptions")
        
        return creative_elements
    
    async def _analyze_storytelling_elements(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze storytelling elements in content"""
        
        text_content = content.get("content", "")
        
        analysis = {
            "has_hook": self._has_compelling_hook(text_content),
            "has_conflict": self._has_conflict_elements(text_content),
            "has_resolution": self._has_resolution_elements(text_content),
            "character_elements": self._count_character_elements(text_content),
            "dialogue_present": self._has_dialogue(text_content),
            "emotional_appeal": self._assess_emotional_appeal(text_content),
            "narrative_flow": self._assess_narrative_flow(text_content)
        }
        
        return analysis
    
    def _has_compelling_hook(self, content: str) -> bool:
        """Check if content has a compelling opening hook"""
        
        # Look for engaging opening patterns
        hook_patterns = [
            r'^[A-Z][^.]*[!?]',  # Starts with exclamation or question
            r'^[A-Z][^.]*imagine',  # Starts with "imagine"
            r'^[A-Z][^.]*what if',  # Starts with "what if"
            r'^[A-Z][^.]*picture this',  # Starts with "picture this"
        ]
        
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in hook_patterns)
    
    def _has_conflict_elements(self, content: str) -> bool:
        """Check if content has conflict elements"""
        
        conflict_words = ['challenge', 'problem', 'struggle', 'difficulty', 'obstacle', 'conflict']
        return any(word in content.lower() for word in conflict_words)
    
    def _has_resolution_elements(self, content: str) -> bool:
        """Check if content has resolution elements"""
        
        resolution_words = ['solution', 'resolve', 'overcome', 'success', 'achieve', 'accomplish']
        return any(word in content.lower() for word in resolution_words)
    
    def _count_character_elements(self, content: str) -> int:
        """Count character-related elements"""
        
        character_indicators = ['he', 'she', 'they', 'person', 'individual', 'character', 'protagonist']
        return sum(content.lower().count(word) for word in character_indicators)
    
    def _has_dialogue(self, content: str) -> bool:
        """Check if content has dialogue"""
        
        return '"' in content or "'" in content
    
    def _assess_emotional_appeal(self, content: str) -> float:
        """Assess emotional appeal of content"""
        
        emotional_words = [
            'amazing', 'incredible', 'fantastic', 'wonderful', 'exciting',
            'inspiring', 'motivating', 'powerful', 'moving', 'touching'
        ]
        
        emotional_count = sum(content.lower().count(word) for word in emotional_words)
        word_count = len(content.split())
        
        return min(1.0, emotional_count / max(word_count, 1) * 10)
    
    def _assess_narrative_flow(self, content: str) -> float:
        """Assess narrative flow of content"""
        
        # Check for transition words
        transition_words = [
            'however', 'therefore', 'meanwhile', 'furthermore', 'additionally',
            'consequently', 'subsequently', 'finally', 'in conclusion'
        ]
        
        transition_count = sum(content.lower().count(word) for word in transition_words)
        sentence_count = len(re.findall(r'[.!?]+', content))
        
        return min(1.0, transition_count / max(sentence_count, 1))
    
    async def _calculate_engagement_metrics(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate engagement metrics for content"""
        
        text_content = content.get("content", "")
        
        metrics = {
            "readability_score": self._calculate_readability_score(text_content),
            "emotional_impact": self._assess_emotional_appeal(text_content),
            "narrative_strength": self._assess_narrative_flow(text_content),
            "interactive_elements": self._count_interactive_elements(text_content),
            "overall_engagement": 0.0
        }
        
        # Calculate overall engagement
        metrics["overall_engagement"] = (
            metrics["readability_score"] * 0.3 +
            metrics["emotional_impact"] * 0.3 +
            metrics["narrative_strength"] * 0.2 +
            metrics["interactive_elements"] * 0.2
        )
        
        return metrics
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score"""
        
        if not content:
            return 0.0
        
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability calculation
        readability = 1.0 - (avg_sentence_length / 20) - (avg_word_length / 10)
        return max(0.0, min(1.0, readability))
    
    def _count_interactive_elements(self, content: str) -> float:
        """Count interactive elements in content"""
        
        interactive_patterns = [
            r'\?',  # Questions
            r'you\b',  # Direct address
            r'your\b',  # Personal pronouns
            r'think about',  # Thinking prompts
            r'consider',  # Consideration prompts
            r'imagine',  # Imagination prompts
        ]
        
        interactive_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in interactive_patterns)
        word_count = len(content.split())
        
        return min(1.0, interactive_count / max(word_count, 1) * 5)
    
    async def _identify_creative_techniques(self, content: Dict[str, Any]) -> List[str]:
        """Identify creative techniques used in content"""
        
        text_content = content.get("content", "")
        techniques = []
        
        # Check for various creative techniques
        if re.search(r'\b(like|as|similar to)\b', text_content, re.IGNORECASE):
            techniques.append("similes")
        
        if re.search(r'\b(is|are|was|were)\b.*\b(like|as)\b', text_content, re.IGNORECASE):
            techniques.append("metaphors")
        
        if re.search(r'\b(imagine|picture|visualize)\b', text_content, re.IGNORECASE):
            techniques.append("visualization")
        
        if re.search(r'\?', text_content):
            techniques.append("rhetorical_questions")
        
        if re.search(r'\b(you|your)\b', text_content, re.IGNORECASE):
            techniques.append("direct_address")
        
        return techniques
    
    async def _assess_readability_enhancements(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Assess readability enhancements"""
        
        text_content = content.get("content", "")
        
        enhancements = {
            "sentence_variety": self._assess_sentence_variety(text_content),
            "paragraph_structure": self._assess_paragraph_structure(text_content),
            "word_choice": self._assess_word_choice(text_content),
            "flow_improvements": self._assess_flow_improvements(text_content)
        }
        
        return enhancements
    
    def _assess_sentence_variety(self, content: str) -> float:
        """Assess sentence variety"""
        
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) < 2:
            return 0.5
        
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        if not sentence_lengths:
            return 0.5
        
        # Calculate variance in sentence lengths
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
        
        # Higher variance = better variety
        return min(1.0, variance / 100)
    
    def _assess_paragraph_structure(self, content: str) -> float:
        """Assess paragraph structure"""
        
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 2:
            return 0.5
        
        # Check for topic sentences and supporting details
        structure_score = 0.0
        
        for paragraph in paragraphs:
            if len(paragraph.split()) > 10:  # Substantial paragraph
                structure_score += 0.1
        
        return min(1.0, structure_score)
    
    def _assess_word_choice(self, content: str) -> float:
        """Assess word choice quality"""
        
        words = content.split()
        if not words:
            return 0.0
        
        # Check for varied vocabulary
        unique_words = len(set(word.lower() for word in words))
        total_words = len(words)
        
        vocabulary_ratio = unique_words / total_words
        
        # Check for sophisticated words
        sophisticated_words = ['consequently', 'furthermore', 'nevertheless', 'moreover', 'therefore']
        sophisticated_count = sum(content.lower().count(word) for word in sophisticated_words)
        
        sophisticated_ratio = sophisticated_count / total_words
        
        return (vocabulary_ratio + sophisticated_ratio) / 2
    
    def _assess_flow_improvements(self, content: str) -> float:
        """Assess flow improvements"""
        
        # Check for transition words
        transition_words = ['however', 'therefore', 'meanwhile', 'furthermore', 'additionally']
        transition_count = sum(content.lower().count(word) for word in transition_words)
        
        # Check for repetition (bad for flow)
        words = content.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        repetition_penalty = sum(1 for count in word_freq.values() if count > 3) / len(words)
        
        flow_score = min(1.0, transition_count / 10) - repetition_penalty
        return max(0.0, flow_score)
    
    async def _calculate_creative_score(self, content: Dict[str, Any]) -> float:
        """Calculate overall creative score"""
        
        text_content = content.get("content", "")
        
        # Base creative elements
        creative_elements = self._extract_creative_elements(text_content)
        element_score = len(creative_elements) / 5  # Normalize to 0-1
        
        # Engagement metrics
        engagement_metrics = await self._calculate_engagement_metrics(content)
        engagement_score = engagement_metrics["overall_engagement"]
        
        # Storytelling elements
        storytelling_elements = await self._analyze_storytelling_elements(content)
        storytelling_score = sum(storytelling_elements.values()) / len(storytelling_elements)
        
        # Overall creative score
        creative_score = (element_score * 0.3 + engagement_score * 0.4 + storytelling_score * 0.3)
        
        return max(0.0, min(1.0, creative_score))
    
    def _calculate_creative_writing_quality_score(
        self,
        output_data: Dict[str, Any],
        task: BaseAgentTask
    ) -> float:
        """Calculate quality score for creative writing"""
        
        base_score = 0.7
        
        # Quality based on creative score
        creative_score = output_data["metadata"].get("creative_score", 0.0)
        base_score += creative_score * 0.2
        
        # Quality based on engagement metrics
        engagement_metrics = output_data.get("engagement_metrics", {})
        overall_engagement = engagement_metrics.get("overall_engagement", 0.0)
        base_score += overall_engagement * 0.1
        
        # Quality based on creative techniques
        creative_techniques = output_data.get("creative_techniques_used", [])
        if len(creative_techniques) > 3:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))

