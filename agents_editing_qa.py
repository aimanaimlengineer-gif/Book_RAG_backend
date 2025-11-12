"""
Editing & QA Agent - Reviews and refines content for quality and consistency
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re

from agents_base_agent import BaseAgent, BaseAgentTask, BaseAgentResult, AgentCapability

logger = logging.getLogger(__name__)

class EditingQAAgent(BaseAgent):
    """Agent responsible for editing, quality assurance, and content refinement"""
    
    def __init__(self, llm_service, config: Dict[str, Any] = None):
        super().__init__(
            agent_type="editing_qa",
            role="Quality Assurance Editor",
            goal="Review, edit, and refine content to ensure high quality, consistency, and readability",
            backstory="You are a senior editor with exceptional attention to detail, expertise in various writing styles, and commitment to excellence. You excel at improving clarity, flow, and overall content quality while maintaining the author's voice.",
            capabilities=[AgentCapability.QUALITY_ASSURANCE, AgentCapability.CONTENT_GENERATION],
            llm_service=llm_service,
            config=config
        )
    
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """Get required capabilities for editing tasks"""
        return [AgentCapability.QUALITY_ASSURANCE]
    
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """Execute editing and QA task"""
        try:
            input_data = task.input_data
            
            # Extract content to edit
            content = input_data.get("content", {})
            illustrations = input_data.get("illustrations", {})
            writing_style = input_data.get("writing_style", "informative")
            target_audience = input_data.get("target_audience", "general")
            
            # Perform comprehensive editing
            edited_content = await self._edit_content(content, writing_style, target_audience)
            
            # Quality assurance checks
            qa_results = await self._perform_quality_assurance(edited_content, writing_style)
            
            # Style and consistency review
            style_review = await self._review_style_consistency(edited_content, writing_style)
            
            # Readability analysis
            readability_analysis = await self._analyze_readability(edited_content, target_audience)
            
            # Grammar and language check
            grammar_review = await self._check_grammar_and_language(edited_content)
            
            # Structure and flow analysis
            structure_analysis = await self._analyze_structure_and_flow(edited_content)
            
            # Generate improvement suggestions
            improvement_suggestions = await self._generate_improvement_suggestions(
                edited_content, qa_results, style_review, readability_analysis
            )
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(
                qa_results, style_review, readability_analysis, grammar_review, structure_analysis
            )
            
            output_data = {
                "edited_content": edited_content,
                "qa_results": qa_results,
                "style_review": style_review,
                "readability_analysis": readability_analysis,
                "grammar_review": grammar_review,
                "structure_analysis": structure_analysis,
                "improvement_suggestions": improvement_suggestions,
                "quality_scores": quality_scores,
                "editorial_metadata": {
                    "edited_by": "EditingQAAgent",
                    "editing_date": datetime.utcnow().isoformat(),
                    "style_guide": writing_style,
                    "target_audience": target_audience,
                    "quality_threshold": self.config.get("quality_threshold", 0.8)
                }
            }
            
            quality_score = self._calculate_editing_quality_score(output_data, task)
            
            return BaseAgentResult(
                success=True,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=0.0,
                output_data=output_data,
                quality_score=quality_score,
                metadata={
                    "chapters_edited": len(edited_content.get("chapters", [])),
                    "overall_quality_score": quality_scores.get("overall_quality", 0.0),
                    "improvements_suggested": len(improvement_suggestions)
                }
            )
            
        except Exception as e:
            logger.error(f"Editing and QA failed: {str(e)}")
            raise
    
    async def _edit_content(
        self,
        content: Dict[str, Any],
        writing_style: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """Edit and refine all content"""
        
        edited_content = content.copy()
        
        # Edit introduction
        if content.get("introduction"):
            edited_content["introduction"] = await self._edit_section(
                content["introduction"], "introduction", writing_style, target_audience
            )
        
        # Edit chapters
        chapters = content.get("chapters", [])
        edited_chapters = []
        
        for chapter in chapters:
            edited_chapter = await self._edit_chapter(chapter, writing_style, target_audience)
            edited_chapters.append(edited_chapter)
        
        edited_content["chapters"] = edited_chapters
        
        # Edit conclusion
        if content.get("conclusion"):
            edited_content["conclusion"] = await self._edit_section(
                content["conclusion"], "conclusion", writing_style, target_audience
            )
        
        # Edit appendices
        appendices = content.get("appendices", [])
        edited_appendices = []
        
        for appendix in appendices:
            edited_appendix = await self._edit_appendix(appendix, writing_style)
            edited_appendices.append(edited_appendix)
        
        edited_content["appendices"] = edited_appendices
        
        return edited_content
    
    async def _edit_chapter(
        self,
        chapter: Dict[str, Any],
        writing_style: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """Edit a single chapter"""
        
        chapter_content = chapter.get("content", "")
        chapter_title = chapter.get("title", "")
        
        editing_prompt = f"""
        Edit this chapter for a {writing_style} book targeting {target_audience}:
        
        Title: {chapter_title}
        Content: {chapter_content}
        
        Improve:
        1. Clarity and readability
        2. Flow and transitions between sections
        3. Consistency with {writing_style} style
        4. Engagement for {target_audience}
        5. Grammar and language precision
        6. Structure and organization
        
        Maintain the original meaning and key points while enhancing quality.
        Return only the edited content, preserving the original structure.
        """
        
        try:
            edited_chapter_content = await self.llm_service.generate_content(
                content="",
                prompt=editing_prompt,
                max_tokens=len(chapter_content.split()) * 2,  # Allow for expansion
                temperature=0.3
            )
            
            edited_chapter = chapter.copy()
            edited_chapter["content"] = edited_chapter_content
            edited_chapter["word_count"] = len(edited_chapter_content.split())
            edited_chapter["edited_at"] = datetime.utcnow().isoformat()
            
            # Update key takeaways if needed
            if chapter.get("key_takeaways"):
                updated_takeaways = await self._update_key_takeaways(
                    edited_chapter_content, chapter.get("key_takeaways", [])
                )
                edited_chapter["key_takeaways"] = updated_takeaways
            
            return edited_chapter
            
        except Exception as e:
            logger.warning(f"Chapter editing failed, returning original: {str(e)}")
            return chapter
    
    async def _edit_section(
        self,
        section_data: Dict[str, Any],
        section_type: str,
        writing_style: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """Edit introduction, conclusion, or other sections"""
        
        section_content = section_data.get("content", "")
        
        editing_prompt = f"""
        Edit this {section_type} section for a {writing_style} book targeting {target_audience}:
        
        {section_content}
        
        For a {section_type}, focus on:
        1. Strong opening/closing impact
        2. Clear purpose and structure
        3. Appropriate tone for {target_audience}
        4. Engaging and memorable language
        5. Smooth flow and coherence
        
        Return the edited content maintaining the original intent.
        """
        
        try:
            edited_content = await self.llm_service.generate_content(
                content="",
                prompt=editing_prompt,
                max_tokens=len(section_content.split()) * 2,
                temperature=0.3
            )
            
            edited_section = section_data.copy()
            edited_section["content"] = edited_content
            edited_section["word_count"] = len(edited_content.split())
            edited_section["edited_at"] = datetime.utcnow().isoformat()
            
            return edited_section
            
        except Exception as e:
            logger.warning(f"Section editing failed: {str(e)}")
            return section_data
    
    async def _edit_appendix(self, appendix: Dict[str, Any], writing_style: str) -> Dict[str, Any]:
        """Edit appendix content"""
        
        appendix_content = appendix.get("content", "")
        appendix_title = appendix.get("title", "")
        
        # Light editing for appendices (mainly formatting and clarity)
        editing_prompt = f"""
        Lightly edit this appendix for clarity and formatting:
        
        Title: {appendix_title}
        Content: {appendix_content}
        
        Focus on:
        1. Clear formatting and structure
        2. Consistency in presentation
        3. Accuracy of information
        4. Proper citations if applicable
        
        Maintain all factual content and references.
        """
        
        try:
            edited_content = await self.llm_service.generate_content(
                content="",
                prompt=editing_prompt,
                max_tokens=len(appendix_content.split()) + 200,
                temperature=0.2
            )
            
            edited_appendix = appendix.copy()
            edited_appendix["content"] = edited_content
            edited_appendix["edited_at"] = datetime.utcnow().isoformat()
            
            return edited_appendix
            
        except Exception as e:
            logger.warning(f"Appendix editing failed: {str(e)}")
            return appendix
    
    async def _perform_quality_assurance(
        self,
        content: Dict[str, Any],
        writing_style: str
    ) -> Dict[str, Any]:
        """Perform comprehensive quality assurance checks"""
        
        qa_results = {
            "content_completeness": {},
            "consistency_check": {},
            "accuracy_review": {},
            "style_adherence": {},
            "overall_qa_score": 0.0
        }
        
        # Check content completeness
        qa_results["content_completeness"] = self._check_content_completeness(content)
        
        # Check consistency across chapters
        qa_results["consistency_check"] = await self._check_consistency(content, writing_style)
        
        # Review accuracy and factual content
        qa_results["accuracy_review"] = await self._review_accuracy(content)
        
        # Check style adherence
        qa_results["style_adherence"] = await self._check_style_adherence(content, writing_style)
        
        # Calculate overall QA score
        scores = [
            qa_results["content_completeness"].get("completeness_score", 0),
            qa_results["consistency_check"].get("consistency_score", 0),
            qa_results["accuracy_review"].get("accuracy_score", 0),
            qa_results["style_adherence"].get("adherence_score", 0)
        ]
        
        qa_results["overall_qa_score"] = sum(scores) / len(scores)
        
        return qa_results
    
    async def _review_style_consistency(
        self,
        content: Dict[str, Any],
        writing_style: str
    ) -> Dict[str, Any]:
        """Review style consistency across content"""
        
        style_review_prompt = f"""
        Review the style consistency across this book content for {writing_style} style:
        
        Introduction present: {bool(content.get('introduction'))}
        Number of chapters: {len(content.get('chapters', []))}
        Conclusion present: {bool(content.get('conclusion'))}
        
        Sample content from first chapter:
        {content.get('chapters', [{}])[0].get('content', '')[:500] if content.get('chapters') else 'No chapters'}...
        
        Evaluate:
        1. Consistency of tone and voice
        2. Adherence to {writing_style} style guidelines
        3. Uniform structure and formatting
        4. Appropriate language level
        5. Consistent terminology usage
        
        Provide style consistency analysis and recommendations.
        Respond in JSON format.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=style_review_prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            style_review = json.loads(response)
            style_review["reviewed_at"] = datetime.utcnow().isoformat()
            
            return style_review
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Style review failed: {str(e)}")
            
            return {
                "tone_consistency": "unknown",
                "style_adherence": "partial",
                "structure_consistency": "unknown",
                "language_level": "appropriate",
                "overall_consistency": "needs_review",
                "notes": "Automated style review failed"
            }
    
    async def _analyze_readability(
        self,
        content: Dict[str, Any],
        target_audience: str
    ) -> Dict[str, Any]:
        """Analyze content readability for target audience"""
        
        # Calculate basic readability metrics
        total_text = ""
        
        if content.get("introduction"):
            total_text += content["introduction"].get("content", "")
        
        for chapter in content.get("chapters", []):
            total_text += chapter.get("content", "")
        
        if content.get("conclusion"):
            total_text += content["conclusion"].get("content", "")
        
        readability_metrics = self._calculate_readability_metrics(total_text)
        
        # LLM analysis for audience appropriateness
        readability_prompt = f"""
        Analyze the readability of this content for {target_audience}:
        
        Content statistics:
        - Total words: {readability_metrics['word_count']}
        - Average sentence length: {readability_metrics['avg_sentence_length']}
        - Average word length: {readability_metrics['avg_word_length']}
        
        Sample text: {total_text[:800]}...
        
        Evaluate appropriateness for {target_audience}:
        1. Vocabulary complexity
        2. Sentence structure
        3. Concept difficulty
        4. Accessibility
        5. Engagement level
        
        Provide readability assessment and suggestions.
        Respond in JSON format.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=readability_prompt,
                max_tokens=600,
                temperature=0.3
            )
            
            readability_analysis = json.loads(response)
            readability_analysis.update(readability_metrics)
            readability_analysis["analyzed_at"] = datetime.utcnow().isoformat()
            
            return readability_analysis
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Readability analysis failed: {str(e)}")
            
            return {
                **readability_metrics,
                "vocabulary_level": "appropriate",
                "sentence_complexity": "moderate",
                "overall_readability": "good",
                "suggestions": ["Manual readability review recommended"]
            }
    
    async def _check_grammar_and_language(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check grammar and language quality"""
        
        grammar_issues = []
        language_quality_score = 0.8  # Base score
        
        # Collect all text for analysis
        all_text = []
        
        if content.get("introduction"):
            all_text.append(content["introduction"].get("content", ""))
        
        for chapter in content.get("chapters", []):
            all_text.append(chapter.get("content", ""))
        
        if content.get("conclusion"):
            all_text.append(content["conclusion"].get("content", ""))
        
        # Basic grammar checks (simplified)
        for text in all_text:
            issues = self._detect_grammar_issues(text)
            grammar_issues.extend(issues)
        
        # Calculate quality score based on issues
        if grammar_issues:
            issue_penalty = min(0.3, len(grammar_issues) * 0.02)
            language_quality_score -= issue_penalty
        
        return {
            "grammar_issues": grammar_issues[:20],  # Limit to top 20 issues
            "total_issues": len(grammar_issues),
            "language_quality_score": max(0.0, language_quality_score),
            "issue_categories": self._categorize_grammar_issues(grammar_issues),
            "checked_at": datetime.utcnow().isoformat()
        }
    
    async def _analyze_structure_and_flow(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content structure and flow"""
        
        structure_prompt = f"""
        Analyze the structure and flow of this book content:
        
        Structure:
        - Introduction: {bool(content.get('introduction'))}
        - Chapters: {len(content.get('chapters', []))}
        - Conclusion: {bool(content.get('conclusion'))}
        - Appendices: {len(content.get('appendices', []))}
        
        Chapter titles:
        {chr(10).join(f"- {ch.get('title', 'Untitled')}" for ch in content.get('chapters', []))}
        
        Evaluate:
        1. Logical progression of topics
        2. Smooth transitions between chapters
        3. Balanced chapter lengths
        4. Clear narrative flow
        5. Effective introduction and conclusion
        
        Provide structure analysis and flow recommendations.
        Respond in JSON format.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=structure_prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            structure_analysis = json.loads(response)
            
            # Add quantitative metrics
            chapter_lengths = [ch.get("word_count", 0) for ch in content.get("chapters", [])]
            
            structure_analysis["quantitative_metrics"] = {
                "chapter_count": len(content.get("chapters", [])),
                "avg_chapter_length": sum(chapter_lengths) / len(chapter_lengths) if chapter_lengths else 0,
                "chapter_length_variance": self._calculate_variance(chapter_lengths),
                "has_introduction": bool(content.get("introduction")),
                "has_conclusion": bool(content.get("conclusion")),
                "structure_completeness": self._calculate_structure_completeness(content)
            }
            
            structure_analysis["analyzed_at"] = datetime.utcnow().isoformat()
            
            return structure_analysis
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Structure analysis failed: {str(e)}")
            
            return {
                "logical_progression": "unknown",
                "transition_quality": "needs_review",
                "chapter_balance": "unknown",
                "narrative_flow": "unclear",
                "overall_structure": "needs_analysis"
            }
    
    async def _generate_improvement_suggestions(
        self,
        content: Dict[str, Any],
        qa_results: Dict[str, Any],
        style_review: Dict[str, Any],
        readability_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific improvement suggestions"""
        
        suggestions = []
        
        # QA-based suggestions
        overall_qa_score = qa_results.get("overall_qa_score", 0.0)
        if overall_qa_score < 0.8:
            suggestions.append({
                "category": "quality_assurance",
                "priority": "high",
                "suggestion": "Address quality assurance issues identified in review",
                "details": "Review completeness, consistency, and accuracy findings"
            })
        
        # Style consistency suggestions
        style_adherence = style_review.get("overall_consistency", "")
        if style_adherence in ["poor", "needs_improvement"]:
            suggestions.append({
                "category": "style_consistency",
                "priority": "medium",
                "suggestion": "Improve style consistency across chapters",
                "details": "Ensure uniform tone, voice, and formatting throughout"
            })
        
        # Readability suggestions
        readability_score = readability_analysis.get("overall_readability", "good")
        if readability_score in ["poor", "difficult"]:
            suggestions.append({
                "category": "readability",
                "priority": "high",
                "suggestion": "Simplify language and structure for target audience",
                "details": "Reduce sentence complexity and use more accessible vocabulary"
            })
        
        # Chapter balance suggestions
        chapters = content.get("chapters", [])
        if chapters:
            chapter_lengths = [ch.get("word_count", 0) for ch in chapters]
            avg_length = sum(chapter_lengths) / len(chapter_lengths)
            
            unbalanced_chapters = [
                i for i, length in enumerate(chapter_lengths)
                if abs(length - avg_length) > avg_length * 0.5
            ]
            
            if unbalanced_chapters:
                suggestions.append({
                    "category": "chapter_balance",
                    "priority": "medium",
                    "suggestion": f"Balance chapter lengths (chapters {unbalanced_chapters} need adjustment)",
                    "details": f"Target average length: {avg_length:.0f} words"
                })
        
        return suggestions
    
    def _check_content_completeness(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all required content sections are present and complete"""
        
        completeness_score = 0.0
        completeness_checks = {}
        
        # Check introduction
        has_introduction = bool(content.get("introduction", {}).get("content"))
        completeness_checks["introduction"] = has_introduction
        if has_introduction:
            completeness_score += 0.2
        
        # Check chapters
        chapters = content.get("chapters", [])
        has_chapters = len(chapters) > 0
        completeness_checks["chapters"] = has_chapters
        if has_chapters:
            completeness_score += 0.5
            
            # Check individual chapters
            complete_chapters = sum(1 for ch in chapters if ch.get("content"))
            chapter_completeness = complete_chapters / len(chapters) if chapters else 0
            completeness_checks["chapter_completeness"] = chapter_completeness
            completeness_score += chapter_completeness * 0.2
        
        # Check conclusion
        has_conclusion = bool(content.get("conclusion", {}).get("content"))
        completeness_checks["conclusion"] = has_conclusion
        if has_conclusion:
            completeness_score += 0.1
        
        return {
            "completeness_score": completeness_score,
            "completeness_checks": completeness_checks,
            "missing_sections": [
                section for section, present in completeness_checks.items()
                if not present and section != "chapter_completeness"
            ]
        }
    
    async def _check_consistency(self, content: Dict[str, Any], writing_style: str) -> Dict[str, Any]:
        """Check consistency across content"""
        
        consistency_issues = []
        consistency_score = 1.0
        
        chapters = content.get("chapters", [])
        
        # Check chapter title consistency
        chapter_titles = [ch.get("title", "") for ch in chapters]
        if len(set(len(title.split()) for title in chapter_titles)) > 3:
            consistency_issues.append("Inconsistent chapter title lengths")
            consistency_score -= 0.1
        
        # Check chapter length consistency
        chapter_lengths = [ch.get("word_count", 0) for ch in chapters]
        if chapter_lengths:
            avg_length = sum(chapter_lengths) / len(chapter_lengths)
            very_different = [
                length for length in chapter_lengths
                if abs(length - avg_length) > avg_length * 0.7
            ]
            if very_different:
                consistency_issues.append(f"Inconsistent chapter lengths: {len(very_different)} chapters significantly different")
                consistency_score -= 0.1
        
        # Check style consistency (basic)
        if len(chapters) > 1:
            first_chapter_style = self._analyze_text_style(chapters[0].get("content", ""))
            style_inconsistencies = 0
            
            for chapter in chapters[1:]:
                chapter_style = self._analyze_text_style(chapter.get("content", ""))
                if self._compare_styles(first_chapter_style, chapter_style) < 0.7:
                    style_inconsistencies += 1
            
            if style_inconsistencies > len(chapters) * 0.3:
                consistency_issues.append("Significant style inconsistencies between chapters")
                consistency_score -= 0.2
        
        return {
            "consistency_score": max(0.0, consistency_score),
            "consistency_issues": consistency_issues,
            "checked_elements": ["chapter_titles", "chapter_lengths", "writing_style"]
        }
    
    async def _review_accuracy(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Review content accuracy (basic check)"""
        
        # This is a basic implementation - in practice, this would integrate
        # with the fact-checking agent results
        
        accuracy_score = 0.8  # Base score assuming good quality
        accuracy_notes = []
        
        # Check for obvious accuracy issues
        all_text = ""
        if content.get("introduction"):
            all_text += content["introduction"].get("content", "")
        
        for chapter in content.get("chapters", []):
            all_text += chapter.get("content", "")
        
        # Basic accuracy indicators
        has_citations = "according to" in all_text.lower() or "source:" in all_text.lower()
        has_statistics = bool(re.search(r'\d+%|\d+ percent', all_text))
        
        if not has_citations and has_statistics:
            accuracy_notes.append("Statistical claims present but citations may be missing")
            accuracy_score -= 0.1
        
        return {
            "accuracy_score": accuracy_score,
            "accuracy_notes": accuracy_notes,
            "has_citations": has_citations,
            "has_statistics": has_statistics,
            "reviewed_at": datetime.utcnow().isoformat()
        }
    
    async def _check_style_adherence(self, content: Dict[str, Any], writing_style: str) -> Dict[str, Any]:
        """Check adherence to specified writing style"""
        
        style_indicators = {
            "technical": ["implementation", "algorithm", "methodology", "procedure"],
            "conversational": ["you", "we", "let's", "imagine"],
            "academic": ["research", "study", "analysis", "evidence"],
            "professional": ["strategy", "approach", "framework", "best practices"],
            "informative": ["information", "details", "explanation", "overview"]
        }
        
        expected_indicators = style_indicators.get(writing_style, [])
        
        # Analyze content for style indicators
        all_text = ""
        for chapter in content.get("chapters", []):
            all_text += chapter.get("content", "").lower()
        
        found_indicators = sum(1 for indicator in expected_indicators if indicator in all_text)
        adherence_score = found_indicators / max(len(expected_indicators), 1)
        
        return {
            "adherence_score": adherence_score,
            "expected_style": writing_style,
            "indicators_found": found_indicators,
            "total_indicators": len(expected_indicators),
            "style_match": "good" if adherence_score > 0.6 else "needs_improvement"
        }
    
    async def _update_key_takeaways(self, content: str, original_takeaways: List[str]) -> List[str]:
        """Update key takeaways based on edited content"""
        
        takeaways_prompt = f"""
        Update these key takeaways based on the edited content:
        
        Original takeaways:
        {chr(10).join(f"- {takeaway}" for takeaway in original_takeaways)}
        
        Updated content: {content[:1000]}...
        
        Provide updated takeaways that accurately reflect the edited content.
        Return as JSON array of strings.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=takeaways_prompt,
                max_tokens=400,
                temperature=0.3
            )
            
            updated_takeaways = json.loads(response)
            return updated_takeaways if isinstance(updated_takeaways, list) else original_takeaways
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Takeaways update failed: {str(e)}")
            return original_takeaways
    
    def _calculate_readability_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate basic readability metrics"""
        
        if not text.strip():
            return {
                "word_count": 0,
                "sentence_count": 0,
                "avg_sentence_length": 0,
                "avg_word_length": 0
            }
        
        # Basic text analysis
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(sentence_count, 1)
        avg_word_length = sum(len(word.strip('.,!?";')) for word in words) / max(word_count, 1)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_word_length": round(avg_word_length, 2)
        }
    
    def _detect_grammar_issues(self, text: str) -> List[Dict[str, Any]]:
        """Detect basic grammar issues (simplified)"""
        
        issues = []
        
        # Basic pattern matching for common issues
        patterns = [
            (r'\b(its)\s+(going|coming)', "Possible confusion: 'its' vs 'it's'"),
            (r'\b(your)\s+(going|coming)', "Possible confusion: 'your' vs 'you're'"),
            (r'\b(there)\s+(going|coming)', "Possible confusion: 'there' vs 'they're'"),
            (r'\s{2,}', "Multiple consecutive spaces"),
            (r'[.]{2,}', "Multiple consecutive periods"),
            (r'^[a-z]', "Sentence may not start with capital letter"),
        ]
        
        sentences = text.split('. ')
        
        for i, sentence in enumerate(sentences):
            for pattern, description in patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    issues.append({
                        "type": "grammar",
                        "description": description,
                        "sentence_index": i,
                        "position": match.start(),
                        "text_snippet": sentence[max(0, match.start()-20):match.end()+20]
                    })
        
        return issues[:50]  # Limit to prevent overwhelming output
    
    def _categorize_grammar_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize grammar issues by type"""
        
        categories = {}
        
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            categories[issue_type] = categories.get(issue_type, 0) + 1
        
        return categories
    
    def _analyze_text_style(self, text: str) -> Dict[str, Any]:
        """Analyze text style characteristics"""
        
        if not text:
            return {}
        
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # Style characteristics
        avg_sentence_length = len(words) / max(len(sentences), 1)
        complex_words = len([w for w in words if len(w) > 7])
        complex_word_ratio = complex_words / max(len(words), 1)
        
        # Tone indicators
        personal_pronouns = len([w for w in words if w.lower() in ['i', 'you', 'we', 'us']])
        personal_ratio = personal_pronouns / max(len(words), 1)
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "complex_word_ratio": complex_word_ratio,
            "personal_pronoun_ratio": personal_ratio,
            "total_words": len(words),
            "total_sentences": len(sentences)
        }
    
    def _compare_styles(self, style1: Dict[str, Any], style2: Dict[str, Any]) -> float:
        """Compare two text styles for similarity"""
        
        if not style1 or not style2:
            return 0.5
        
        # Compare key metrics
        metrics = ["avg_sentence_length", "complex_word_ratio", "personal_pronoun_ratio"]
        similarity_scores = []
        
        for metric in metrics:
            val1 = style1.get(metric, 0)
            val2 = style2.get(metric, 0)
            
            if val1 == 0 and val2 == 0:
                similarity = 1.0
            else:
                max_val = max(val1, val2)
                min_val = min(val1, val2)
                similarity = min_val / max(max_val, 0.001)  # Avoid division by zero
            
            similarity_scores.append(similarity)
        
        return sum(similarity_scores) / len(similarity_scores)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        return variance
    
    def _calculate_structure_completeness(self, content: Dict[str, Any]) -> float:
        """Calculate structure completeness score"""
        
        completeness = 0.0
        
        # Check required sections
        if content.get("introduction"):
            completeness += 0.2
        
        chapters = content.get("chapters", [])
        if chapters:
            completeness += 0.6
            
            # Bonus for complete chapters
            complete_chapters = sum(1 for ch in chapters if ch.get("content") and len(ch.get("content", "").split()) > 100)
            chapter_bonus = (complete_chapters / len(chapters)) * 0.1
            completeness += chapter_bonus
        
        if content.get("conclusion"):
            completeness += 0.2
        
        return completeness
    
    def _calculate_quality_scores(
        self,
        qa_results: Dict[str, Any],
        style_review: Dict[str, Any],
        readability_analysis: Dict[str, Any],
        grammar_review: Dict[str, Any],
        structure_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality scores"""
        
        scores = {
            "qa_score": qa_results.get("overall_qa_score", 0.0),
            "style_score": 0.8,  # Default if no specific score available
            "readability_score": 0.8,  # Default
            "grammar_score": grammar_review.get("language_quality_score", 0.0),
            "structure_score": structure_analysis.get("quantitative_metrics", {}).get("structure_completeness", 0.0)
        }
        
        # Parse style and readability scores if available
        if isinstance(style_review.get("overall_consistency"), str):
            style_map = {"excellent": 1.0, "good": 0.8, "fair": 0.6, "poor": 0.4}
            scores["style_score"] = style_map.get(style_review["overall_consistency"], 0.6)
        
        if isinstance(readability_analysis.get("overall_readability"), str):
            readability_map = {"excellent": 1.0, "good": 0.8, "fair": 0.6, "poor": 0.4}
            scores["readability_score"] = readability_map.get(readability_analysis["overall_readability"], 0.6)
        
        # Calculate overall quality
        scores["overall_quality"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _calculate_editing_quality_score(
        self,
        output_data: Dict[str, Any],
        task: BaseAgentTask
    ) -> float:
        """Calculate quality score for editing process"""
        
        quality_scores = output_data.get("quality_scores", {})
        overall_quality = quality_scores.get("overall_quality", 0.0)
        
        # Base score from content quality
        base_score = overall_quality
        
        # Bonus for comprehensive editing
        if len(output_data.get("improvement_suggestions", [])) > 0:
            base_score += 0.05
        
        # Bonus for thorough analysis
        if output_data.get("readability_analysis") and output_data.get("grammar_review"):
            base_score += 0.05
        
        return max(0.0, min(1.0, base_score))