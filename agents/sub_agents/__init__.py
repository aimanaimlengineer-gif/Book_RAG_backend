"""
Sub-Agents Module - Specialized agents for enhanced functionality
"""

from .web_research_sub_agent import WebResearchSubAgent
from .academic_research_sub_agent import AcademicResearchSubAgent
from .source_verification_sub_agent import SourceVerificationSubAgent
from .content_verification_sub_agent import ContentVerificationSubAgent
from .language_processing_sub_agent import LanguageProcessingSubAgent
from .creative_writing_sub_agent import CreativeWritingSubAgent
from .style_consistency_sub_agent import StyleConsistencySubAgent
from .visual_design_sub_agent import VisualDesignSubAgent
from .grammar_syntax_sub_agent import GrammarSyntaxSubAgent

__all__ = [
    'WebResearchSubAgent',
    'AcademicResearchSubAgent', 
    'SourceVerificationSubAgent',
    'ContentVerificationSubAgent',
    'LanguageProcessingSubAgent',
    'CreativeWritingSubAgent',
    'StyleConsistencySubAgent',
    'VisualDesignSubAgent',
    'GrammarSyntaxSubAgent'
]
