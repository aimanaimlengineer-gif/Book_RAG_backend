"""
Illustration & Visual Agent - Generates visual content and diagrams for books
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re
import base64
import io

from agents_base_agent import BaseAgent, BaseAgentTask, BaseAgentResult, AgentCapability

logger = logging.getLogger(__name__)

class IllustrationAgent(BaseAgent):
    """Agent responsible for generating illustrations, diagrams, and visual content"""
    
    def __init__(self, llm_service, image_generation_service=None, config: Dict[str, Any] = None):
        self.image_generation_service = image_generation_service
        super().__init__(
            agent_type="illustration",
            role="Visual Content Creator",
            goal="Generate high-quality illustrations, diagrams, and visual elements that enhance book content",
            backstory="You are an expert visual designer and illustrator with deep understanding of educational graphics, technical diagrams, and engaging visual storytelling. You excel at creating visuals that clarify complex concepts.",
            capabilities=[AgentCapability.VISUAL_CONTENT, AgentCapability.CONTENT_GENERATION],
            llm_service=llm_service,
            config=config
        )
    
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """Get required capabilities for illustration tasks"""
        return [AgentCapability.VISUAL_CONTENT]
    
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """Execute illustration generation task"""
        try:
            input_data = task.input_data
            
            # Extract content for illustration
            content = input_data.get("content", {})
            category = input_data.get("category", "")
            writing_style = input_data.get("writing_style", "informative")
            target_audience = input_data.get("target_audience", "general")
            
            # Analyze content for illustration opportunities
            illustration_plan = await self._analyze_content_for_illustrations(
                content, category, writing_style, target_audience
            )
            
            # Generate illustrations based on plan
            generated_illustrations = await self._generate_illustrations(
                illustration_plan, content, category
            )
            
            # Create diagram specifications
            diagram_specifications = await self._create_diagram_specifications(
                content, category
            )
            
            # Generate infographics
            infographics = await self._generate_infographics(
                content, category, target_audience
            )
            
            # Create visual content metadata
            visual_metadata = self._create_visual_metadata(
                generated_illustrations, diagram_specifications, infographics
            )
            
            output_data = {
                "illustration_plan": illustration_plan,
                "generated_illustrations": generated_illustrations,
                "diagram_specifications": diagram_specifications,
                "infographics": infographics,
                "visual_metadata": visual_metadata,
                "integration_instructions": await self._create_integration_instructions(
                    generated_illustrations, content
                ),
                "alt_text_descriptions": await self._generate_alt_text_descriptions(
                    generated_illustrations
                )
            }
            
            quality_score = self._calculate_illustration_quality_score(output_data, task)
            
            return BaseAgentResult(
                success=True,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=0.0,
                output_data=output_data,
                quality_score=quality_score,
                metadata={
                    "illustrations_generated": len(generated_illustrations),
                    "diagrams_specified": len(diagram_specifications),
                    "infographics_created": len(infographics),
                    "total_visual_elements": len(generated_illustrations) + len(diagram_specifications) + len(infographics)
                }
            )
            
        except Exception as e:
            logger.error(f"Illustration generation failed: {str(e)}")
            raise
    
    async def _analyze_content_for_illustrations(
        self,
        content: Dict[str, Any],
        category: str,
        writing_style: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """Analyze content to identify illustration opportunities"""
        
        analysis_prompt = f"""
        Analyze this {category} book content to identify opportunities for visual elements:
        
        Writing Style: {writing_style}
        Target Audience: {target_audience}
        
        Chapter Information:
        {self._format_content_for_analysis(content)}
        
        Identify opportunities for:
        1. Conceptual illustrations (abstract concepts made visual)
        2. Process diagrams (step-by-step workflows)
        3. Data visualizations (charts, graphs, statistics)
        4. Technical diagrams (system architecture, relationships)
        5. Infographics (summary information)
        6. Character/mascot illustrations
        7. Decorative elements
        
        For each opportunity, specify:
        - Type of visual
        - Chapter/section location
        - Purpose and learning objective
        - Complexity level
        - Priority (high/medium/low)
        
        Respond in JSON format with structured recommendations.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=analysis_prompt,
                max_tokens=1500,
                temperature=0.4
            )
            
            illustration_plan = json.loads(response)
            
            # Add metadata
            illustration_plan["analysis_metadata"] = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "content_type": category,
                "analysis_method": "llm_content_analysis",
                "total_opportunities": len(illustration_plan.get("opportunities", []))
            }
            
            return illustration_plan
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Content analysis failed, using fallback: {str(e)}")
            return self._create_fallback_illustration_plan(content, category)
    
    async def _generate_illustrations(
        self,
        illustration_plan: Dict[str, Any],
        content: Dict[str, Any],
        category: str
    ) -> List[Dict[str, Any]]:
        """Generate actual illustrations based on the plan"""
        
        generated_illustrations = []
        opportunities = illustration_plan.get("opportunities", [])
        
        # Prioritize high and medium priority illustrations
        priority_opportunities = [
            opp for opp in opportunities 
            if opp.get("priority", "low") in ["high", "medium"]
        ]
        
        for opportunity in priority_opportunities[:10]:  # Limit to prevent overwhelming
            try:
                illustration = await self._create_single_illustration(
                    opportunity, content, category
                )
                generated_illustrations.append(illustration)
                
            except Exception as e:
                logger.warning(f"Failed to generate illustration: {str(e)}")
                
                # Create placeholder illustration
                placeholder = self._create_placeholder_illustration(opportunity)
                generated_illustrations.append(placeholder)
        
        return generated_illustrations
    
    async def _create_single_illustration(
        self,
        opportunity: Dict[str, Any],
        content: Dict[str, Any],
        category: str
    ) -> Dict[str, Any]:
        """Create a single illustration"""
        
        illustration_type = opportunity.get("type", "conceptual")
        chapter_location = opportunity.get("chapter_location", 1)
        purpose = opportunity.get("purpose", "")
        
        # Generate detailed description for the illustration
        description_prompt = f"""
        Create a detailed visual description for this {category} book illustration:
        
        Type: {illustration_type}
        Purpose: {purpose}
        Chapter: {chapter_location}
        
        Context from content:
        {self._get_chapter_context(content, chapter_location)}
        
        Provide:
        1. Detailed visual description (style, composition, colors)
        2. Specific elements to include
        3. Text or labels needed
        4. Mood and tone
        5. Technical specifications
        
        Make it suitable for {category} content and educational purposes.
        Respond in JSON format.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=description_prompt,
                max_tokens=800,
                temperature=0.6
            )
            
            illustration_description = json.loads(response)
            
            # Generate the actual image if service is available
            image_data = None
            if self.image_generation_service:
                try:
                    image_data = await self._generate_image(illustration_description)
                except Exception as e:
                    logger.warning(f"Image generation failed: {str(e)}")
            
            illustration = {
                "id": f"ill_{len(content.get('chapters', []))}{chapter_location}_{opportunity.get('type', 'img')}",
                "type": illustration_type,
                "chapter_location": chapter_location,
                "purpose": purpose,
                "description": illustration_description,
                "image_data": image_data,
                "placeholder_needed": image_data is None,
                "file_format": "png" if image_data else "placeholder",
                "dimensions": {"width": 800, "height": 600},
                "created_at": datetime.utcnow().isoformat()
            }
            
            return illustration
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Illustration creation failed: {str(e)}")
            return self._create_placeholder_illustration(opportunity)
    
    async def _create_diagram_specifications(
        self,
        content: Dict[str, Any],
        category: str
    ) -> List[Dict[str, Any]]:
        """Create specifications for technical diagrams"""
        
        diagram_prompt = f"""
        Identify and specify technical diagrams needed for this {category} book:
        
        Content Overview:
        {self._format_content_for_analysis(content)}
        
        Consider diagrams for:
        1. Process flows and workflows
        2. System architectures
        3. Hierarchical relationships
        4. Cause-and-effect relationships
        5. Timelines and sequences
        6. Comparison matrices
        
        For each diagram, provide:
        - Diagram type (flowchart, org chart, timeline, etc.)
        - Chapter and section
        - Elements to include
        - Connections and relationships
        - Labels and annotations
        - Complexity level
        
        Respond in JSON format with diagram specifications.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=diagram_prompt,
                max_tokens=1200,
                temperature=0.3
            )
            
            diagram_specs = json.loads(response)
            
            # Process and format diagram specifications
            formatted_specs = []
            for spec in diagram_specs.get("diagrams", []):
                formatted_spec = {
                    "id": f"diag_{spec.get('chapter', 1)}_{spec.get('type', 'flow')}",
                    "type": spec.get("type", "flowchart"),
                    "chapter": spec.get("chapter", 1),
                    "section": spec.get("section", ""),
                    "title": spec.get("title", ""),
                    "elements": spec.get("elements", []),
                    "connections": spec.get("connections", []),
                    "labels": spec.get("labels", []),
                    "complexity": spec.get("complexity", "medium"),
                    "creation_method": "mermaid" if spec.get("type") in ["flowchart", "sequence", "gantt"] else "manual",
                    "mermaid_code": self._generate_mermaid_code(spec) if spec.get("type") in ["flowchart", "sequence"] else None,
                    "created_at": datetime.utcnow().isoformat()
                }
                formatted_specs.append(formatted_spec)
            
            return formatted_specs
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Diagram specification failed: {str(e)}")
            return self._create_fallback_diagrams(content, category)
    
    async def _generate_infographics(
        self,
        content: Dict[str, Any],
        category: str,
        target_audience: str
    ) -> List[Dict[str, Any]]:
        """Generate infographic specifications"""
        
        infographic_prompt = f"""
        Design infographics for this {category} book targeting {target_audience}:
        
        Content Summary:
        {self._format_content_for_analysis(content)}
        
        Create infographics for:
        1. Key statistics and data points
        2. Process summaries
        3. Comparison charts
        4. Timeline visualizations
        5. Quick reference guides
        
        For each infographic, specify:
        - Title and purpose
        - Data/information to visualize
        - Visual style and layout
        - Color scheme suggestions
        - Text hierarchy
        - Call-out elements
        
        Make them engaging for {target_audience}.
        Respond in JSON format.
        """
        
        try:
            response = await self.llm_service.generate_content(
                content="",
                prompt=infographic_prompt,
                max_tokens=1000,
                temperature=0.5
            )
            
            infographic_specs = json.loads(response)
            
            # Format infographic specifications
            formatted_infographics = []
            for spec in infographic_specs.get("infographics", []):
                formatted_infographic = {
                    "id": f"info_{spec.get('chapter', 1)}_{len(formatted_infographics)}",
                    "title": spec.get("title", ""),
                    "purpose": spec.get("purpose", ""),
                    "chapter": spec.get("chapter", 1),
                    "data_points": spec.get("data_points", []),
                    "visual_style": spec.get("visual_style", "modern"),
                    "layout": spec.get("layout", "vertical"),
                    "color_scheme": spec.get("color_scheme", ["#3498db", "#2ecc71", "#e74c3c"]),
                    "dimensions": {"width": 800, "height": 1000},
                    "complexity": spec.get("complexity", "medium"),
                    "created_at": datetime.utcnow().isoformat()
                }
                formatted_infographics.append(formatted_infographic)
            
            return formatted_infographics
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Infographic generation failed: {str(e)}")
            return []
    
    async def _generate_image(self, illustration_description: Dict[str, Any]) -> Optional[str]:
        """Generate actual image using image generation service"""
        
        if not self.image_generation_service:
            return None
        
        # Create prompt for image generation
        visual_prompt = illustration_description.get("visual_description", "")
        style = illustration_description.get("style", "professional illustration")
        
        full_prompt = f"{visual_prompt}, {style}, high quality, educational, clean design"
        
        try:
            # This would integrate with actual image generation service
            # For now, return a placeholder indication
            generated_image = await self.image_generation_service.generate_image(
                prompt=full_prompt,
                width=800,
                height=600,
                style="illustration"
            )
            
            return generated_image  # Base64 encoded image data
            
        except Exception as e:
            logger.error(f"Image generation service failed: {str(e)}")
            return None
    
    async def _create_integration_instructions(
        self,
        illustrations: List[Dict[str, Any]],
        content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create instructions for integrating visuals into content"""
        
        integration_instructions = {
            "placement_guidelines": {},
            "chapter_integrations": {},
            "formatting_instructions": {},
            "responsive_design": {}
        }
        
        # Group illustrations by chapter
        illustrations_by_chapter = {}
        for illustration in illustrations:
            chapter = illustration.get("chapter_location", 1)
            if chapter not in illustrations_by_chapter:
                illustrations_by_chapter[chapter] = []
            illustrations_by_chapter[chapter].append(illustration)
        
        # Create integration instructions for each chapter
        for chapter_num, chapter_illustrations in illustrations_by_chapter.items():
            chapter_data = self._get_chapter_data(content, chapter_num)
            
            integration_instructions["chapter_integrations"][f"chapter_{chapter_num}"] = {
                "total_illustrations": len(chapter_illustrations),
                "placement_suggestions": await self._suggest_placements(
                    chapter_illustrations, chapter_data
                ),
                "spacing_guidelines": "Minimum 2 paragraphs between illustrations",
                "caption_style": "Below image, italic text, 12pt font"
            }
        
        # General formatting instructions
        integration_instructions["formatting_instructions"] = {
            "image_alignment": "center",
            "max_width": "80% of page width",
            "margin": "20px top and bottom",
            "border": "1px solid #e0e0e0",
            "alt_text": "Required for all images",
            "caption_format": "Figure X.Y: Description"
        }
        
        # Responsive design guidelines
        integration_instructions["responsive_design"] = {
            "mobile_max_width": "100%",
            "tablet_max_width": "90%",
            "desktop_max_width": "80%",
            "quality_scaling": "Use high-DPI images for crisp display"
        }
        
        return integration_instructions
    
    async def _generate_alt_text_descriptions(
        self,
        illustrations: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate alt text descriptions for accessibility"""
        
        alt_texts = {}
        
        for illustration in illustrations:
            illustration_id = illustration.get("id", "")
            purpose = illustration.get("purpose", "")
            description = illustration.get("description", {})
            
            alt_text_prompt = f"""
            Create accessible alt text for this illustration:
            
            Purpose: {purpose}
            Visual Description: {description.get('visual_description', '')}
            Elements: {description.get('elements', [])}
            
            Create concise, descriptive alt text (under 125 characters) that conveys:
            1. What the image shows
            2. Its purpose/function
            3. Key information it provides
            
            Focus on meaning, not artistic details.
            """
            
            try:
                alt_text = await self.llm_service.generate_content(
                    content="",
                    prompt=alt_text_prompt,
                    max_tokens=100,
                    temperature=0.2
                )
                
                # Clean and truncate alt text
                clean_alt_text = alt_text.strip().replace('"', '').replace('\n', ' ')[:125]
                alt_texts[illustration_id] = clean_alt_text
                
            except Exception as e:
                logger.warning(f"Alt text generation failed for {illustration_id}: {str(e)}")
                alt_texts[illustration_id] = f"Illustration showing {purpose}"
        
        return alt_texts
    
    async def _suggest_placements(
        self,
        chapter_illustrations: List[Dict[str, Any]],
        chapter_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest optimal placements for illustrations within chapter"""
        
        placement_suggestions = []
        chapter_content = chapter_data.get("content", "")
        
        # Analyze chapter structure
        paragraphs = [p.strip() for p in chapter_content.split('\n\n') if p.strip()]
        total_paragraphs = len(paragraphs)
        
        for i, illustration in enumerate(chapter_illustrations):
            # Distribute illustrations evenly throughout chapter
            suggested_paragraph = int((i + 1) * total_paragraphs / (len(chapter_illustrations) + 1))
            
            placement_suggestions.append({
                "illustration_id": illustration.get("id"),
                "suggested_position": f"After paragraph {suggested_paragraph}",
                "context_paragraph": paragraphs[min(suggested_paragraph - 1, len(paragraphs) - 1)][:100] + "..." if paragraphs else "",
                "placement_rationale": f"Supports content around paragraph {suggested_paragraph}",
                "alternative_positions": [
                    f"After paragraph {max(1, suggested_paragraph - 1)}",
                    f"After paragraph {min(total_paragraphs, suggested_paragraph + 1)}"
                ]
            })
        
        return placement_suggestions
    
    def _format_content_for_analysis(self, content: Dict[str, Any]) -> str:
        """Format content for LLM analysis"""
        
        formatted_content = ""
        
        chapters = content.get("chapters", [])
        for i, chapter in enumerate(chapters[:5]):  # Limit to first 5 chapters
            formatted_content += f"Chapter {chapter.get('chapter_number', i+1)}: {chapter.get('title', 'Untitled')}\n"
            
            # Add key concepts and summary
            key_concepts = chapter.get("key_concepts", [])
            if key_concepts:
                formatted_content += f"Key Concepts: {', '.join(key_concepts[:5])}\n"
            
            # Add content preview
            chapter_content = chapter.get("content", "")
            if chapter_content:
                formatted_content += f"Content Preview: {chapter_content[:300]}...\n"
            
            formatted_content += "\n"
        
        return formatted_content
    
    def _get_chapter_context(self, content: Dict[str, Any], chapter_location: int) -> str:
        """Get context from specific chapter"""
        
        chapters = content.get("chapters", [])
        
        for chapter in chapters:
            if chapter.get("chapter_number") == chapter_location:
                chapter_content = chapter.get("content", "")
                return chapter_content[:500] + "..." if len(chapter_content) > 500 else chapter_content
        
        return "Chapter context not available"
    
    def _get_chapter_data(self, content: Dict[str, Any], chapter_num: int) -> Dict[str, Any]:
        """Get full chapter data"""
        
        chapters = content.get("chapters", [])
        
        for chapter in chapters:
            if chapter.get("chapter_number") == chapter_num:
                return chapter
        
        return {}
    
    def _generate_mermaid_code(self, diagram_spec: Dict[str, Any]) -> str:
        """Generate Mermaid diagram code"""
        
        diagram_type = diagram_spec.get("type", "flowchart")
        elements = diagram_spec.get("elements", [])
        connections = diagram_spec.get("connections", [])
        
        if diagram_type == "flowchart":
            mermaid_code = "flowchart TD\n"
            
            # Add elements
            for element in elements:
                element_id = element.get("id", "")
                element_label = element.get("label", "")
                element_shape = element.get("shape", "rect")
                
                if element_shape == "rect":
                    mermaid_code += f"    {element_id}[{element_label}]\n"
                elif element_shape == "diamond":
                    mermaid_code += f"    {element_id}{{{element_label}}}\n"
                elif element_shape == "circle":
                    mermaid_code += f"    {element_id}(({element_label}))\n"
            
            # Add connections
            for connection in connections:
                from_id = connection.get("from", "")
                to_id = connection.get("to", "")
                label = connection.get("label", "")
                
                if label:
                    mermaid_code += f"    {from_id} -->|{label}| {to_id}\n"
                else:
                    mermaid_code += f"    {from_id} --> {to_id}\n"
            
            return mermaid_code
        
        elif diagram_type == "sequence":
            mermaid_code = "sequenceDiagram\n"
            
            # Add participants
            participants = set()
            for connection in connections:
                participants.add(connection.get("from", ""))
                participants.add(connection.get("to", ""))
            
            for participant in participants:
                mermaid_code += f"    participant {participant}\n"
            
            # Add interactions
            for connection in connections:
                from_participant = connection.get("from", "")
                to_participant = connection.get("to", "")
                message = connection.get("label", "message")
                
                mermaid_code += f"    {from_participant}->>{to_participant}: {message}\n"
            
            return mermaid_code
        
        return "graph TD\n    A[Diagram] --> B[Not Implemented]"
    
    def _create_fallback_illustration_plan(
        self,
        content: Dict[str, Any],
        category: str
    ) -> Dict[str, Any]:
        """Create fallback illustration plan when LLM analysis fails"""
        
        chapters = content.get("chapters", [])
        opportunities = []
        
        # Create basic opportunities for each chapter
        for i, chapter in enumerate(chapters[:5]):  # Limit to 5 chapters
            chapter_num = chapter.get("chapter_number", i + 1)
            chapter_title = chapter.get("title", f"Chapter {chapter_num}")
            
            opportunities.append({
                "type": "conceptual",
                "chapter_location": chapter_num,
                "purpose": f"Illustrate key concepts from {chapter_title}",
                "priority": "medium",
                "complexity": "medium"
            })
            
            # Add diagram opportunity for technical categories
            if category in ["technology", "business"]:
                opportunities.append({
                    "type": "diagram",
                    "chapter_location": chapter_num,
                    "purpose": f"Process diagram for {chapter_title}",
                    "priority": "low",
                    "complexity": "medium"
                })
        
        return {
            "opportunities": opportunities,
            "analysis_metadata": {
                "analyzed_at": datetime.utcnow().isoformat(),
                "content_type": category,
                "analysis_method": "fallback_basic",
                "total_opportunities": len(opportunities)
            }
        }
    
    def _create_placeholder_illustration(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Create placeholder illustration when generation fails"""
        
        return {
            "id": f"placeholder_{opportunity.get('type', 'img')}_{opportunity.get('chapter_location', 1)}",
            "type": opportunity.get("type", "placeholder"),
            "chapter_location": opportunity.get("chapter_location", 1),
            "purpose": opportunity.get("purpose", "Visual placeholder"),
            "description": {
                "visual_description": f"Placeholder for {opportunity.get('type', 'illustration')}",
                "elements": ["Placeholder content"],
                "style": "Simple placeholder design"
            },
            "image_data": None,
            "placeholder_needed": True,
            "file_format": "placeholder",
            "dimensions": {"width": 800, "height": 600},
            "created_at": datetime.utcnow().isoformat(),
            "notes": "Generated as placeholder due to creation failure"
        }
    
    def _create_fallback_diagrams(
        self,
        content: Dict[str, Any],
        category: str
    ) -> List[Dict[str, Any]]:
        """Create fallback diagrams when specification fails"""
        
        fallback_diagrams = []
        
        # Add basic process diagram for each category
        category_diagrams = {
            "technology": {
                "type": "flowchart",
                "title": "Development Process",
                "elements": [
                    {"id": "A", "label": "Planning", "shape": "rect"},
                    {"id": "B", "label": "Development", "shape": "rect"},
                    {"id": "C", "label": "Testing", "shape": "rect"},
                    {"id": "D", "label": "Deployment", "shape": "rect"}
                ],
                "connections": [
                    {"from": "A", "to": "B"},
                    {"from": "B", "to": "C"},
                    {"from": "C", "to": "D"}
                ]
            },
            "business": {
                "type": "flowchart",
                "title": "Business Process",
                "elements": [
                    {"id": "A", "label": "Strategy", "shape": "rect"},
                    {"id": "B", "label": "Planning", "shape": "rect"},
                    {"id": "C", "label": "Execution", "shape": "rect"},
                    {"id": "D", "label": "Review", "shape": "rect"}
                ],
                "connections": [
                    {"from": "A", "to": "B"},
                    {"from": "B", "to": "C"},
                    {"from": "C", "to": "D"},
                    {"from": "D", "to": "A", "label": "feedback"}
                ]
            }
        }
        
        if category in category_diagrams:
            diagram_template = category_diagrams[category]
            
            fallback_diagrams.append({
                "id": f"diag_fallback_{category}",
                "type": diagram_template["type"],
                "chapter": 1,
                "section": "Overview",
                "title": diagram_template["title"],
                "elements": diagram_template["elements"],
                "connections": diagram_template["connections"],
                "labels": [],
                "complexity": "simple",
                "creation_method": "mermaid",
                "mermaid_code": self._generate_mermaid_code(diagram_template),
                "created_at": datetime.utcnow().isoformat(),
                "notes": "Fallback diagram template"
            })
        
        return fallback_diagrams
    
    def _create_visual_metadata(
        self,
        illustrations: List[Dict[str, Any]],
        diagrams: List[Dict[str, Any]],
        infographics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create metadata for all visual content"""
        
        return {
            "total_visual_elements": len(illustrations) + len(diagrams) + len(infographics),
            "illustrations_count": len(illustrations),
            "diagrams_count": len(diagrams),
            "infographics_count": len(infographics),
            "visual_distribution": {
                f"chapter_{i+1}": len([
                    v for v in (illustrations + diagrams + infographics)
                    if v.get("chapter_location", v.get("chapter", 1)) == i+1
                ])
                for i in range(10)  # Assuming max 10 chapters
            },
            "complexity_distribution": {
                "simple": len([v for v in (illustrations + diagrams + infographics) if v.get("complexity") == "simple"]),
                "medium": len([v for v in (illustrations + diagrams + infographics) if v.get("complexity") == "medium"]),
                "complex": len([v for v in (illustrations + diagrams + infographics) if v.get("complexity") == "complex"])
            },
            "generation_summary": {
                "generated_at": datetime.utcnow().isoformat(),
                "successful_generations": len([i for i in illustrations if not i.get("placeholder_needed", False)]),
                "placeholder_count": len([i for i in illustrations if i.get("placeholder_needed", False)]),
                "diagram_specifications": len(diagrams),
                "infographic_designs": len(infographics)
            }
        }
    
    def _calculate_illustration_quality_score(
        self,
        output_data: Dict[str, Any],
        task: BaseAgentTask
    ) -> float:
        """Calculate quality score for illustration generation"""
        
        base_score = 0.7
        
        # Quality based on number of visual elements generated
        total_visuals = output_data["visual_metadata"]["total_visual_elements"]
        if total_visuals >= 5:
            base_score += 0.1
        elif total_visuals >= 3:
            base_score += 0.05
        
        # Quality based on successful generation vs placeholders
        successful_generations = output_data["visual_metadata"]["generation_summary"]["successful_generations"]
        total_illustrations = output_data["visual_metadata"]["illustrations_count"]
        
        if total_illustrations > 0:
            success_ratio = successful_generations / total_illustrations
            base_score += success_ratio * 0.1
        
        # Quality based on comprehensive coverage
        if (output_data.get("diagram_specifications") and 
            output_data.get("integration_instructions") and
            output_data.get("alt_text_descriptions")):
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))