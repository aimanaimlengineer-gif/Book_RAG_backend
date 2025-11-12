"""
Content Generation Agent - Creates engaging and well-structured book content
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re

from agents_base_agent import BaseAgent, BaseAgentTask, BaseAgentResult, AgentCapability

logger = logging.getLogger(__name__)

class ContentGenerationAgent(BaseAgent):
    """Agent responsible for generating high-quality book content"""
    
    def __init__(self, llm_service, embedding_service, config: Dict[str, Any] = None):
        self.embedding_service = embedding_service
        super().__init__(
            agent_type="content_generation",
            role="Content Creation Writer",
            goal="Generate engaging, well-structured, and educational book content based on research and outlines",
            backstory="You are a professional writer and content creator with expertise in educational materials, engaging storytelling, and clear communication. You excel at transforming research into compelling, accessible content.",
            capabilities=[AgentCapability.CONTENT_GENERATION, AgentCapability.CREATIVE_WRITING],
            llm_service=llm_service,
            config=config
        )
    
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """Get required capabilities for content generation tasks"""
        return [AgentCapability.CONTENT_GENERATION]
    
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """Execute content generation task"""
        try:
            input_data = task.input_data
            
            # Extract task parameters
            verified_research = input_data.get("verified_research", {})
            chapter_outline = input_data.get("chapter_outline", [])
            writing_style = input_data.get("writing_style", "informative")
            target_audience = input_data.get("target_audience", "general")
            language = input_data.get("language", "en")
            
            # Generate content for each chapter
            generated_chapters = await self._generate_all_chapters(
                chapter_outline, verified_research, writing_style, target_audience, language
            )
            
            # Create introduction and conclusion
            introduction = await self._generate_introduction(
                input_data.get("title", ""), chapter_outline, writing_style, target_audience
            )
            
            conclusion = await self._generate_conclusion(
                generated_chapters, writing_style, target_audience
            )
            
            # Generate supplementary content
            appendices = await self._generate_appendices(verified_research)
            
            # Calculate content statistics
            content_stats = self._calculate_content_statistics(
                introduction, generated_chapters, conclusion, appendices
            )
            
            output_data = {
                "introduction": introduction,
                "chapters": generated_chapters,
                "conclusion": conclusion,
                "appendices": appendices,
                "content_statistics": content_stats,
                "writing_metadata": {
                    "style": writing_style,
                    "audience": target_audience,
                    "language": language,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
            quality_score = self._calculate_content_quality_score(output_data, task)
            
            return BaseAgentResult(
                success=True,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=0.0,
                output_data=output_data,
                quality_score=quality_score,
                metadata={
                    "total_word_count": content_stats.get("total_words", 0),
                    "chapters_generated": len(generated_chapters),
                    "avg_chapter_length": content_stats.get("avg_chapter_length", 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise
    
    async def _generate_all_chapters(
        self,
        chapter_outline: List[Dict[str, Any]],
        verified_research: Dict[str, Any],
        writing_style: str,
        target_audience: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """Generate content for all chapters"""
        
        generated_chapters = []
        
        for chapter_info in chapter_outline:
            try:
                chapter_content = await self._generate_single_chapter(
                    chapter_info, verified_research, writing_style, target_audience, language
                )
                generated_chapters.append(chapter_content)
                
                # Small delay between chapters to avoid overwhelming the LLM
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to generate chapter {chapter_info.get('chapter_number', '?')}: {str(e)}")
                # Generate fallback chapter
                fallback_chapter = await self._generate_fallback_chapter(chapter_info, writing_style)
                generated_chapters.append(fallback_chapter)
        
        return generated_chapters
    
    async def _generate_single_chapter(
        self,
        chapter_info: Dict[str, Any],
        verified_research: Dict[str, Any],
        writing_style: str,
        target_audience: str,
        language: str
    ) -> Dict[str, Any]:
        """Generate content for a single chapter"""
        
        chapter_number = chapter_info.get("chapter_number", 1)
        chapter_title = chapter_info.get("title", f"Chapter {chapter_number}")
        learning_objectives = chapter_info.get("learning_objectives", [])
        key_concepts = chapter_info.get("key_concepts", [])
        target_word_count = chapter_info.get("estimated_word_count", 2500)
        
        # Get relevant research for this chapter
        chapter_research = self._extract_chapter_research(chapter_number, verified_research)
        
        # Generate chapter content prompt
        content_prompt = f"""
        Write Chapter {chapter_number}: "{chapter_title}" for a {writing_style} book targeting {target_audience}.
        
        Chapter Requirements:
        - Target word count: {target_word_count} words
        - Language: {language}
        - Writing style: {writing_style}
        
        Learning Objectives:
        {chr(10).join(f"- {obj}" for obj in learning_objectives)}
        
        Key Concepts to Cover:
        {chr(10).join(f"- {concept}" for concept in key_concepts)}
        
        Available Research:
        {json.dumps(chapter_research, indent=2)}
        
        Structure the chapter with:
        1. Engaging introduction that hooks the reader
        2. Clear explanation of key concepts with examples
        3. Practical applications and real-world connections
        4. Step-by-step guidance where appropriate
        5. Summary of key takeaways
        6. Smooth transition to next chapter
        
        Write in an engaging, educational tone that makes complex concepts accessible.
        Include concrete examples and actionable insights throughout.
        
        Provide the complete chapter content in well-structured paragraphs.
        """
        
        try:
            # Generate main chapter content
            chapter_content = await self.llm_service.generate_content(
                content="",
                prompt=content_prompt,
                max_tokens=int(target_word_count * 1.5),  # Allow for some flexibility
                temperature=0.7
            )
            
            # Generate chapter exercises (if appropriate for writing style)
            exercises = await self._generate_chapter_exercises(
                chapter_info, writing_style, target_audience
            )
            
            # Generate key takeaways
            key_takeaways = await self._generate_key_takeaways(
                chapter_content, key_concepts
            )
            
            return {
                "chapter_number": chapter_number,
                "title": chapter_title,
                "content": chapter_content,
                "exercises": exercises,
                "key_takeaways": key_takeaways,
                "word_count": len(chapter_content.split()),
                "learning_objectives_covered": learning_objectives,
                "research_sources_used": len(chapter_research.get("sources", [])),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate chapter content: {str(e)}")
            raise
    
    async def _generate_chapter_exercises(
        self,
        chapter_info: Dict[str, Any],
        writing_style: str,
        target_audience: str
    ) -> List[Dict[str, Any]]:
        """Generate exercises and activities for the chapter"""
        
        if writing_style in ["technical", "educational", "academic"]:
            exercise_prompt = f"""
            Create 3-5 practical exercises for Chapter {chapter_info.get('chapter_number')}: "{chapter_info.get('title')}"
            
            Target audience: {target_audience}
            Key concepts: {', '.join(chapter_info.get('key_concepts', []))}
            
            For each exercise, provide:
            1. Exercise title
            2. Objective (what the reader will learn)
            3. Instructions (step-by-step)
            4. Expected outcome
            5. Difficulty level (beginner/intermediate/advanced)
            
            Make exercises practical and directly applicable to the chapter content.
            Provide response in JSON format.
            """
            
            try:
                response = await self.llm_service.generate_content(
                    content="",
                    prompt=exercise_prompt,
                    max_tokens=800,
                    temperature=0.6
                )
                
                exercises_data = json.loads(response)
                return exercises_data.get("exercises", [])
                
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Exercise generation failed: {str(e)}")
                return []
        
        return []
    
    async def _generate_key_takeaways(
        self,
        chapter_content: str,
        key_concepts: List[str]
    ) -> List[str]:
        """Generate key takeaways from chapter content"""
        
        takeaways_prompt = f"""
        Extract 5-7 key takeaways from this chapter content:
        
        Key Concepts: {', '.join(key_concepts)}
        
        Chapter Content: {chapter_content[:2000]}...
        
        Provide clear, actionable takeaways that readers can remember and apply.
        Each takeaway should be a single, concise sentence.
        Return as a JSON array of strings.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=takeaways_prompt,
                max_tokens=400,
                temperature=0.3
            )
            
            takeaways_data = json.loads(response)
            return takeaways_data if isinstance(takeaways_data, list) else []
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Takeaways generation failed: {str(e)}")
            # Fallback: extract from key concepts
            return [f"Understanding {concept} is essential for practical application" for concept in key_concepts[:5]]
    
    async def _generate_introduction(
        self,
        book_title: str,
        chapter_outline: List[Dict[str, Any]],
        writing_style: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """Generate book introduction"""
        
        # Extract book overview from chapter outline
        chapter_titles = [ch.get("title", "") for ch in chapter_outline]
        total_chapters = len(chapter_outline)
        
        intro_prompt = f"""
        Write an engaging introduction for the book "{book_title}" targeting {target_audience}.
        
        Writing style: {writing_style}
        
        The book contains {total_chapters} chapters:
        {chr(10).join(f"Chapter {i+1}: {title}" for i, title in enumerate(chapter_titles))}
        
        The introduction should:
        1. Hook the reader with a compelling opening
        2. Clearly explain what the book covers and why it matters
        3. Describe who the book is for and what they'll gain
        4. Provide a roadmap of the chapters
        5. Set expectations for the reading experience
        6. Motivate the reader to continue
        
        Target length: 800-1200 words
        
        Write in an engaging, accessible tone that builds excitement for the content ahead.
        """
        
        try:
            intro_content = await self.llm_service.generate_content(
                content="",
                prompt=intro_prompt,
                max_tokens=1500,
                temperature=0.7
            )
            
            return {
                "content": intro_content,
                "word_count": len(intro_content.split()),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Introduction generation failed: {str(e)}")
            return {
                "content": f"Welcome to {book_title}. This comprehensive guide will take you through essential concepts and practical applications.",
                "word_count": 20,
                "generated_at": datetime.utcnow().isoformat(),
                "note": "Fallback introduction due to generation failure"
            }
    
    async def _generate_conclusion(
        self,
        generated_chapters: List[Dict[str, Any]],
        writing_style: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """Generate book conclusion"""
        
        # Extract key themes from chapters
        all_takeaways = []
        for chapter in generated_chapters:
            takeaways = chapter.get("key_takeaways", [])
            all_takeaways.extend(takeaways)
        
        conclusion_prompt = f"""
        Write a compelling conclusion for a {writing_style} book targeting {target_audience}.
        
        Key themes covered in the book:
        {chr(10).join(f"- {takeaway}" for takeaway in all_takeaways[:10])}
        
        The conclusion should:
        1. Summarize the main insights and learnings
        2. Reinforce the key messages
        3. Provide next steps for readers
        4. Inspire action and continued learning
        5. End on a motivating and memorable note
        
        Target length: 600-900 words
        
        Write in an inspiring tone that empowers readers to apply what they've learned.
        """
        
        try:
            conclusion_content = await self.llm_service.generate_content(
                content="",
                prompt=conclusion_prompt,
                max_tokens=1200,
                temperature=0.7
            )
            
            return {
                "content": conclusion_content,
                "word_count": len(conclusion_content.split()),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Conclusion generation failed: {str(e)}")
            return {
                "content": "Thank you for reading. Apply these insights to achieve your goals and continue your learning journey.",
                "word_count": 17,
                "generated_at": datetime.utcnow().isoformat(),
                "note": "Fallback conclusion due to generation failure"
            }
    
    async def _generate_appendices(self, verified_research: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate appendices and supplementary materials"""
        
        appendices = []
        
        # Glossary
        glossary = await self._generate_glossary(verified_research)
        if glossary:
            appendices.append({
                "title": "Glossary",
                "type": "glossary",
                "content": glossary
            })
        
        # Resources and further reading
        resources = self._compile_resources(verified_research)
        if resources:
            appendices.append({
                "title": "Additional Resources",
                "type": "resources",
                "content": resources
            })
        
        # Citations and references
        citations = verified_research.get("citations", [])
        if citations:
            appendices.append({
                "title": "References",
                "type": "references",
                "content": self._format_citations(citations)
            })
        
        return appendices
    
    async def _generate_glossary(self, verified_research: Dict[str, Any]) -> Optional[str]:
        """Generate glossary of key terms"""
        
        # Extract technical terms from research
        knowledge_base = verified_research.get("knowledge_base", {})
        structured_knowledge = verified_research.get("structured_knowledge", {})
        
        # Collect terms from all chapters
        all_concepts = []
        for chapter_data in structured_knowledge.values():
            concepts = chapter_data.get("key_concepts", [])
            all_concepts.extend(concepts)
        
        if not all_concepts:
            return None
        
        # Remove duplicates and sort
        unique_concepts = sorted(list(set(all_concepts)))
        
        glossary_prompt = f"""
        Create a glossary for these key terms: {', '.join(unique_concepts[:20])}
        
        For each term, provide:
        - Clear, concise definition
        - Context for how it's used in the book
        - Keep definitions accessible to the target audience
        
        Format as a simple list with term followed by definition.
        """
        
        try:
            glossary_content = await self.llm_service.generate_content(
                content="",
                prompt=glossary_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            return glossary_content
            
        except Exception as e:
            logger.warning(f"Glossary generation failed: {str(e)}")
            return None
    
    def _compile_resources(self, verified_research: Dict[str, Any]) -> Optional[str]:
        """Compile additional resources for readers"""
        
        knowledge_base = verified_research.get("knowledge_base", {})
        sources = knowledge_base.get("sources", [])
        
        if not sources:
            return None
        
        # Group sources by type
        web_resources = [s for s in sources if s.get("source_type") == "web" and s.get("credibility_score", 0) > 0.7]
        
        resources_text = "**Recommended Reading and Resources**\n\n"
        
        for i, source in enumerate(web_resources[:10], 1):
            title = source.get("title", "Unnamed Resource")
            url = source.get("url", "")
            description = source.get("content", "")[:200] + "..." if source.get("content") else ""
            
            resources_text += f"{i}. **{title}**\n"
            if url:
                resources_text += f"   URL: {url}\n"
            if description:
                resources_text += f"   {description}\n\n"
        
        return resources_text
    
    def _format_citations(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations for references section"""
        
        references_text = "**References**\n\n"
        
        for citation in citations:
            formatted_citation = citation.get("formatted_citation", "")
            if formatted_citation:
                references_text += f"{citation.get('id', '')}. {formatted_citation}\n\n"
        
        return references_text
    
    def _extract_chapter_research(self, chapter_number: int, verified_research: Dict[str, Any]) -> Dict[str, Any]:
        """Extract research relevant to specific chapter"""
        
        structured_knowledge = verified_research.get("structured_knowledge", {})
        chapter_key = f"chapter_{chapter_number}"
        
        return structured_knowledge.get(chapter_key, {})
    
    async def _generate_fallback_chapter(
        self,
        chapter_info: Dict[str, Any],
        writing_style: str
    ) -> Dict[str, Any]:
        """Generate fallback chapter when main generation fails"""
        
        chapter_number = chapter_info.get("chapter_number", 1)
        chapter_title = chapter_info.get("title", f"Chapter {chapter_number}")
        key_concepts = chapter_info.get("key_concepts", [])
        
        fallback_content = f"""
        # {chapter_title}
        
        This chapter covers essential concepts related to {', '.join(key_concepts[:3]) if key_concepts else 'the topic'}.
        
        ## Introduction
        
        In this chapter, we'll explore the fundamental principles and practical applications that form the foundation of understanding.
        
        ## Key Concepts
        
        {chr(10).join(f"- {concept}" for concept in key_concepts) if key_concepts else "- Core principles and best practices"}
        
        ## Summary
        
        This chapter has provided an overview of essential concepts and their practical applications.
        """
        
        return {
            "chapter_number": chapter_number,
            "title": chapter_title,
            "content": fallback_content,
            "exercises": [],
            "key_takeaways": key_concepts[:3] if key_concepts else ["Understanding core concepts is essential"],
            "word_count": len(fallback_content.split()),
            "learning_objectives_covered": chapter_info.get("learning_objectives", []),
            "research_sources_used": 0,
            "generated_at": datetime.utcnow().isoformat(),
            "note": "Fallback content due to generation failure"
        }
    
    def _calculate_content_statistics(
        self,
        introduction: Dict[str, Any],
        chapters: List[Dict[str, Any]],
        conclusion: Dict[str, Any],
        appendices: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive content statistics"""
        
        intro_words = introduction.get("word_count", 0)
        conclusion_words = conclusion.get("word_count", 0)
        
        chapter_words = [ch.get("word_count", 0) for ch in chapters]
        total_chapter_words = sum(chapter_words)
        
        appendix_words = sum(len(app.get("content", "").split()) for app in appendices)
        
        total_words = intro_words + total_chapter_words + conclusion_words + appendix_words
        
        return {
            "total_words": total_words,
            "introduction_words": intro_words,
            "chapter_words": total_chapter_words,
            "conclusion_words": conclusion_words,
            "appendix_words": appendix_words,
            "avg_chapter_length": total_chapter_words // len(chapters) if chapters else 0,
            "chapter_count": len(chapters),
            "longest_chapter": max(chapter_words) if chapter_words else 0,
            "shortest_chapter": min(chapter_words) if chapter_words else 0,
            "estimated_pages": total_words // 250  # Assuming 250 words per page
        }
    
    def _calculate_content_quality_score(
        self,
        output_data: Dict[str, Any],
        task: BaseAgentTask
    ) -> float:
        """Calculate quality score for generated content"""
        
        base_score = 0.7
        
        # Check word count targets
        content_stats = output_data.get("content_statistics", {})
        target_length = task.input_data.get("target_length", 15000)
        actual_length = content_stats.get("total_words", 0)
        
        # Quality bonus for meeting length targets (within 20% tolerance)
        if 0.8 * target_length <= actual_length <= 1.2 * target_length:
            base_score += 0.1
        
        # Quality bonus for chapter consistency
        chapters = output_data.get("chapters", [])
        if chapters:
            chapter_lengths = [ch.get("word_count", 0) for ch in chapters]
            avg_length = sum(chapter_lengths) / len(chapter_lengths)
            
            # Check consistency (standard deviation)
            variance = sum((length - avg_length) ** 2 for length in chapter_lengths) / len(chapter_lengths)
            std_dev = variance ** 0.5
            
            if std_dev < avg_length * 0.3:  # Within 30% of average
                base_score += 0.1
        
        # Quality bonus for complete structure
        if (output_data.get("introduction") and 
            output_data.get("conclusion") and 
            len(chapters) > 0):
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))