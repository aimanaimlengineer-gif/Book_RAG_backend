"""
Category Selection Agent - Analyzes user preferences and selects book category
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from agents_base_agent import BaseAgent, BaseAgentTask, BaseAgentResult, AgentCapability
from config_settings import BOOK_CATEGORIES

logger = logging.getLogger(__name__)

class CategorySelectionAgent(BaseAgent):
    """Agent responsible for category selection based on user preferences"""
    
    def __init__(self, llm_service, config: Dict[str, Any] = None):
        super().__init__(
            agent_type="category_selection",
            role="Category Selection Specialist",
            goal="Analyze user preferences and select the most appropriate book category with high accuracy",
            backstory="You are an expert in content categorization with deep understanding of different book genres, target audiences, and market preferences. You excel at matching user intentions with appropriate content categories.",
            capabilities=[AgentCapability.CATEGORY_ANALYSIS],
            llm_service=llm_service,
            config=config
        )
    
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """Get required capabilities for category selection tasks"""
        return [AgentCapability.CATEGORY_ANALYSIS]
    
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """Execute category selection task"""
        try:
            input_data = task.input_data
            user_preferences = input_data.get("user_preferences", {})
            title = input_data.get("title", "")
            target_audience = input_data.get("target_audience", "")
            custom_requirements = input_data.get("custom_requirements", "")
            
            # Analyze preferences and select category
            category_analysis = await self._analyze_category_preferences(
                title, user_preferences, target_audience, custom_requirements
            )
            
            selected_category = category_analysis["selected_category"]
            confidence_score = category_analysis["confidence_score"]
            
            # Get category configuration
            category_config = BOOK_CATEGORIES.get(selected_category, {})
            
            # Validate selection meets minimum confidence threshold
            min_confidence = task.config.get("min_confidence", 0.8)
            if confidence_score < min_confidence:
                logger.warning(f"Low confidence category selection: {confidence_score}")
            
            output_data = {
                "selected_category": selected_category,
                "confidence_score": confidence_score,
                "category_config": category_config,
                "analysis": category_analysis["analysis"],
                "recommendations": category_analysis["recommendations"],
                "alternative_categories": category_analysis.get("alternatives", [])
            }
            
            quality_score = self._calculate_category_quality_score(
                confidence_score, category_analysis, task
            )
            
            return BaseAgentResult(
                success=True,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=0.0,  # Will be set by base class
                output_data=output_data,
                quality_score=quality_score,
                metadata={
                    "categories_considered": list(BOOK_CATEGORIES.keys()),
                    "analysis_method": "llm_preference_matching"
                }
            )
            
        except Exception as e:
            logger.error(f"Category selection failed: {str(e)}")
            raise
    
    async def _analyze_category_preferences(
        self, 
        title: str, 
        user_preferences: Dict[str, Any], 
        target_audience: str, 
        custom_requirements: str
    ) -> Dict[str, Any]:
        """Analyze user input to determine the best category"""
        
        # Prepare analysis prompt
        categories_info = self._format_categories_for_analysis()
        
        analysis_prompt = f"""
        Analyze the following information to select the most appropriate book category:
        
        Title: {title}
        Target Audience: {target_audience}
        User Preferences: {user_preferences}
        Custom Requirements: {custom_requirements}
        
        Available Categories:
        {categories_info}
        
        Please provide:
        1. The most suitable category (use exact category key)
        2. Confidence score (0.0 to 1.0)
        3. Detailed analysis explaining your choice
        4. Specific recommendations for this category
        5. Alternative categories if confidence is below 0.9
        
        Respond in JSON format with keys: selected_category, confidence_score, analysis, recommendations, alternatives
        """
        
        try:
            # Get LLM analysis
            response = await self.llm_service.generate_content(
                content="",
                prompt=analysis_prompt,
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse response
            import json
            analysis_result = json.loads(response)
            
            # Validate selected category
            selected_category = analysis_result.get("selected_category")
            if selected_category not in BOOK_CATEGORIES:
                # Fallback to best match
                selected_category = self._find_best_category_match(title, user_preferences)
                analysis_result["selected_category"] = selected_category
                analysis_result["confidence_score"] = 0.6  # Lower confidence for fallback
            
            return analysis_result
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM analysis failed, using fallback: {str(e)}")
            return self._fallback_category_selection(title, user_preferences, target_audience)
    
    def _format_categories_for_analysis(self) -> str:
        """Format category information for LLM analysis"""
        categories_text = ""
        for key, info in BOOK_CATEGORIES.items():
            categories_text += f"""
            {key}:
            - Name: {info['name']}
            - Description: {info['description']}
            - Style: {info['style']}
            - Target Length: {info['target_length']} pages
            - Chapter Count: {info['chapter_count']}
            """
        return categories_text
    
    def _find_best_category_match(
        self, 
        title: str, 
        user_preferences: Dict[str, Any]
    ) -> str:
        """Fallback method to find best category match using simple heuristics"""
        title_lower = title.lower()
        
        # Keyword-based matching
        if any(word in title_lower for word in ['tech', 'programming', 'software', 'code', 'ai', 'ml']):
            return 'technology'
        elif any(word in title_lower for word in ['business', 'startup', 'entrepreneur', 'management']):
            return 'business'
        elif any(word in title_lower for word in ['health', 'fitness', 'wellness', 'mental']):
            return 'health'
        elif any(word in title_lower for word in ['finance', 'money', 'investment', 'financial']):
            return 'finance'
        elif any(word in title_lower for word in ['education', 'learning', 'study', 'academic']):
            return 'education'
        else:
            return 'lifestyle'  # Default fallback
    
    def _fallback_category_selection(
        self, 
        title: str, 
        user_preferences: Dict[str, Any], 
        target_audience: str
    ) -> Dict[str, Any]:
        """Fallback category selection when LLM fails"""
        selected_category = self._find_best_category_match(title, user_preferences)
        
        return {
            "selected_category": selected_category,
            "confidence_score": 0.5,
            "analysis": "Fallback selection based on keyword matching due to analysis failure",
            "recommendations": f"Consider refining the title and preferences for better categorization",
            "alternatives": [cat for cat in BOOK_CATEGORIES.keys() if cat != selected_category][:3]
        }
    
    def _calculate_category_quality_score(
        self, 
        confidence_score: float, 
        analysis: Dict[str, Any], 
        task: BaseAgentTask
    ) -> float:
        """Calculate quality score for category selection"""
        base_score = confidence_score
        
        # Bonus for detailed analysis
        if len(analysis.get("analysis", "")) > 100:
            base_score += 0.1
        
        # Bonus for providing alternatives
        if analysis.get("alternatives") and len(analysis["alternatives"]) > 0:
            base_score += 0.05
        
        # Bonus for specific recommendations
        if analysis.get("recommendations") and len(analysis["recommendations"]) > 50:
            base_score += 0.05
        
        return max(0.0, min(1.0, base_score))

    async def validate_category_selection(
        self, 
        book_project_id: str, 
        category: str, 
        user_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and potentially revise category selection based on user feedback"""
        task = BaseAgentTask(
            book_project_id=book_project_id,
            agent_type=self.agent_type,
            task_name="category_validation",
            input_data={
                "current_category": category,
                "user_feedback": user_feedback,
                "available_categories": list(BOOK_CATEGORIES.keys())
            }
        )
        
        result = await self.execute_task(task)
        return result.output_data if result.success else {}
    
    async def suggest_category_improvements(
        self, 
        book_project_id: str, 
        current_category: str, 
        book_progress: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest category improvements based on book development progress"""
        
        improvement_prompt = f"""
        Based on the book development progress, analyze if the current category selection is still optimal:
        
        Current Category: {current_category}
        Book Progress: {book_progress}
        
        Consider:
        1. Has the content direction changed from initial conception?
        2. Is the target audience still aligned with the category?
        3. Would a different category better serve the content?
        4. Are there emerging themes that suggest a category shift?
        
        Provide recommendations for category optimization or confirmation.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=improvement_prompt,
                max_tokens=800,
                temperature=0.4
            )
            
            return {
                "current_category": current_category,
                "recommendations": response,
                "requires_change": "category change recommended" in response.lower(),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Category improvement analysis failed: {str(e)}")
            return {
                "current_category": current_category,
                "error": str(e),
                "recommendations": "Unable to analyze category improvements at this time"
            }