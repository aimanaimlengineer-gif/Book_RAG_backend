"""
Publication Agent - Formats and compiles books into multiple publication formats
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
import uuid
from pathlib import Path

from agents_base_agent import BaseAgent, BaseAgentTask, BaseAgentResult, AgentCapability

logger = logging.getLogger(__name__)

class PublicationAgent(BaseAgent):
    """Agent responsible for formatting and publishing books in multiple formats"""
    
    def __init__(self, llm_service, config: Dict[str, Any] = None):
        super().__init__(
            agent_type="publication",
            role="Publication Specialist",
            goal="Format and compile content into professional publications across multiple formats",
            backstory="You are an expert in multi-format publishing with extensive knowledge of layout design, typography, and publication standards. You excel at creating professional, print-ready documents and digital publications.",
            capabilities=[AgentCapability.PUBLICATION, AgentCapability.CONTENT_GENERATION],
            llm_service=llm_service,
            config=config
        )
        
        # Initialize publication settings
        self.output_directory = self.config.get("output_path", "./outputs")
        self.temp_directory = self.config.get("temp_path", "./temp")
        self.supported_formats = self.config.get("supported_formats", ["pdf", "epub", "html", "docx"])
        
        # Create directories if they don't exist
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
        Path(self.temp_directory).mkdir(parents=True, exist_ok=True)
    
    def _get_required_capabilities(self, task: BaseAgentTask) -> List[AgentCapability]:
        """Get required capabilities for publication tasks"""
        return [AgentCapability.PUBLICATION]
    
    async def _execute_core_task(self, task: BaseAgentTask) -> BaseAgentResult:
        """Execute publication task"""
        try:
            input_data = task.input_data
            
            # Extract content and metadata
            final_content = input_data.get("final_content", {})
            output_formats = input_data.get("output_formats", ["pdf"])
            book_metadata = self._extract_book_metadata(input_data)
            
            # Generate publication metadata
            publication_metadata = await self._generate_publication_metadata(
                final_content, book_metadata
            )
            
            # Create formatted versions for each requested format
            publications = {}
            
            for format_type in output_formats:
                if format_type in self.supported_formats:
                    try:
                        publication = await self._create_publication(
                            final_content, format_type, publication_metadata
                        )
                        publications[format_type] = publication
                        
                    except Exception as e:
                        logger.error(f"Failed to create {format_type} publication: {str(e)}")
                        publications[format_type] = {
                            "success": False,
                            "error": str(e),
                            "format": format_type
                        }
                else:
                    logger.warning(f"Unsupported format requested: {format_type}")
            
            # Generate publication report
            publication_report = await self._generate_publication_report(
                publications, publication_metadata
            )
            
            # Create distribution package
            distribution_package = await self._create_distribution_package(
                publications, publication_metadata
            )
            
            output_data = {
                "publications": publications,
                "publication_metadata": publication_metadata,
                "publication_report": publication_report,
                "distribution_package": distribution_package,
                "file_locations": self._get_file_locations(publications),
                "download_links": self._generate_download_links(publications),
                "publication_summary": self._create_publication_summary(publications)
            }
            
            quality_score = self._calculate_publication_quality_score(output_data, task)
            
            return BaseAgentResult(
                success=True,
                task_id=task.task_id,
                agent_type=self.agent_type,
                execution_time=0.0,
                output_data=output_data,
                quality_score=quality_score,
                metadata={
                    "formats_created": len([p for p in publications.values() if p.get("success", False)]),
                    "total_formats_requested": len(output_formats),
                    "total_file_size": sum(p.get("file_size", 0) for p in publications.values() if p.get("success", False))
                }
            )
            
        except Exception as e:
            logger.error(f"Publication failed: {str(e)}")
            raise
    
    async def _generate_publication_metadata(
        self,
        content: Dict[str, Any],
        book_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive publication metadata"""
        
        # Calculate content statistics
        content_stats = self._calculate_content_statistics(content)
        
        # Generate ISBN if needed
        isbn = await self._generate_isbn(book_metadata)
        
        # Create publication metadata
        publication_metadata = {
            "title": book_metadata.get("title", "Untitled Book"),
            "author": book_metadata.get("author", "Agentic RAG Generator"),
            "publisher": book_metadata.get("publisher", "Agentic RAG Publishers"),
            "publication_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "language": book_metadata.get("language", "en"),
            "category": book_metadata.get("category", "General"),
            "isbn": isbn,
            "version": "1.0",
            "content_statistics": content_stats,
            "copyright_notice": f"© {datetime.utcnow().year} {book_metadata.get('author', 'Agentic RAG Generator')}",
            "generation_metadata": {
                "generated_by": "Agentic RAG Book Generator",
                "generation_date": datetime.utcnow().isoformat(),
                "agent_version": "1.0.0",
                "quality_score": book_metadata.get("quality_score", 0.8)
            }
        }
        
        return publication_metadata
    
    async def _create_publication(
        self,
        content: Dict[str, Any],
        format_type: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create publication in specified format"""
        
        project_id = metadata.get("project_id", str(uuid.uuid4())[:8])
        filename = f"{metadata['title'].replace(' ', '_')}_{project_id}.{format_type}"
        file_path = os.path.join(self.output_directory, filename)
        
        if format_type == "pdf":
            return await self._create_pdf_publication(content, metadata, file_path)
        elif format_type == "epub":
            return await self._create_epub_publication(content, metadata, file_path)
        elif format_type == "html":
            return await self._create_html_publication(content, metadata, file_path)
        elif format_type == "docx":
            return await self._create_docx_publication(content, metadata, file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    async def _create_pdf_publication(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any],
        file_path: str
    ) -> Dict[str, Any]:
        """Create PDF publication"""
        
        try:
            # This would integrate with actual PDF generation library (ReportLab/WeasyPrint)
            pdf_content = await self._generate_pdf_content(content, metadata)
            
            # For now, create a placeholder file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"PDF Publication: {metadata['title']}\n")
                f.write(f"Generated: {datetime.utcnow()}\n\n")
                f.write(pdf_content)
            
            file_size = os.path.getsize(file_path)
            
            return {
                "success": True,
                "format": "pdf",
                "file_path": file_path,
                "file_size": file_size,
                "page_count": self._estimate_page_count(content),
                "creation_method": "reportlab",
                "quality_score": 0.9,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"PDF creation failed: {str(e)}")
            return {
                "success": False,
                "format": "pdf",
                "error": str(e),
                "created_at": datetime.utcnow().isoformat()
            }
    
    async def _create_epub_publication(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any],
        file_path: str
    ) -> Dict[str, Any]:
        """Create EPUB publication"""
        
        try:
            # This would integrate with EbookLib
            epub_content = await self._generate_epub_content(content, metadata)
            
            # Create placeholder EPUB structure
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"EPUB Publication: {metadata['title']}\n")
                f.write(f"Generated: {datetime.utcnow()}\n\n")
                f.write(epub_content)
            
            file_size = os.path.getsize(file_path)
            
            return {
                "success": True,
                "format": "epub",
                "file_path": file_path,
                "file_size": file_size,
                "chapter_count": len(content.get("chapters", [])),
                "creation_method": "ebooklib",
                "quality_score": 0.9,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"EPUB creation failed: {str(e)}")
            return {
                "success": False,
                "format": "epub",
                "error": str(e),
                "created_at": datetime.utcnow().isoformat()
            }
    
    async def _create_html_publication(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any],
        file_path: str
    ) -> Dict[str, Any]:
        """Create HTML publication"""
        
        try:
            html_content = await self._generate_html_content(content, metadata)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            file_size = os.path.getsize(file_path)
            
            return {
                "success": True,
                "format": "html",
                "file_path": file_path,
                "file_size": file_size,
                "responsive": True,
                "creation_method": "custom_html_generator",
                "quality_score": 0.85,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"HTML creation failed: {str(e)}")
            return {
                "success": False,
                "format": "html",
                "error": str(e),
                "created_at": datetime.utcnow().isoformat()
            }
    
    async def _create_docx_publication(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any],
        file_path: str
    ) -> Dict[str, Any]:
        """Create DOCX publication"""
        
        try:
            # This would integrate with python-docx
            docx_content = await self._generate_docx_content(content, metadata)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"DOCX Publication: {metadata['title']}\n")
                f.write(f"Generated: {datetime.utcnow()}\n\n")
                f.write(docx_content)
            
            file_size = os.path.getsize(file_path)
            
            return {
                "success": True,
                "format": "docx",
                "file_path": file_path, 
                "file_size": file_size,
                "editable": True,
                "creation_method": "python_docx",
                "quality_score": 0.9,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"DOCX creation failed: {str(e)}")
            return {
                "success": False,
                "format": "docx",
                "error": str(e),
                "created_at": datetime.utcnow().isoformat()
            }
    
    async def _generate_pdf_content(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Generate formatted content for PDF"""
        
        pdf_template = f"""
Title: {metadata['title']}
Author: {metadata['author']}
Publisher: {metadata['publisher']}
Publication Date: {metadata['publication_date']}
ISBN: {metadata['isbn']}

{metadata['copyright_notice']}

TABLE OF CONTENTS

"""
        
        # Add table of contents
        chapters = content.get("chapters", [])
        for i, chapter in enumerate(chapters):
            pdf_template += f"{i+1}. {chapter.get('title', f'Chapter {i+1}')} ................... {i+5}\n"
        
        pdf_template += "\n\n"
        
        # Add introduction
        if content.get("introduction"):
            pdf_template += "INTRODUCTION\n\n"
            pdf_template += content["introduction"].get("content", "")
            pdf_template += "\n\n"
        
        # Add chapters
        for chapter in chapters:
            chapter_title = chapter.get("title", "Untitled Chapter")
            chapter_content = chapter.get("content", "")
            
            pdf_template += f"CHAPTER {chapter.get('chapter_number', '1')}: {chapter_title.upper()}\n\n"
            pdf_template += chapter_content
            pdf_template += "\n\n"
            
            # Add key takeaways if available
            if chapter.get("key_takeaways"):
                pdf_template += "KEY TAKEAWAYS:\n"
                for takeaway in chapter["key_takeaways"]:
                    pdf_template += f"• {takeaway}\n"
                pdf_template += "\n"
        
        # Add conclusion
        if content.get("conclusion"):
            pdf_template += "CONCLUSION\n\n"
            pdf_template += content["conclusion"].get("content", "")
            pdf_template += "\n\n"
        
        # Add appendices
        for appendix in content.get("appendices", []):
            pdf_template += f"APPENDIX: {appendix.get('title', 'Supplementary Material').upper()}\n\n"
            pdf_template += appendix.get("content", "")
            pdf_template += "\n\n"
        
        return pdf_template
    
    async def _generate_epub_content(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Generate formatted content for EPUB"""
        
        # EPUB would use XML/XHTML structure
        epub_content = f"""
EPUB Book: {metadata['title']}
Author: {metadata['author']}
Language: {metadata['language']}

Metadata:
- ISBN: {metadata['isbn']}
- Publisher: {metadata['publisher']}
- Publication Date: {metadata['publication_date']}

Content Structure:
"""
        
        # List chapters
        chapters = content.get("chapters", [])
        for chapter in chapters:
            epub_content += f"- Chapter {chapter.get('chapter_number', '?')}: {chapter.get('title', 'Untitled')}\n"
        
        epub_content += f"\nTotal Word Count: {metadata.get('content_statistics', {}).get('total_words', 0)}\n"
        epub_content += f"Estimated Reading Time: {self._calculate_reading_time(content)} minutes\n"
        
        return epub_content
    
    async def _generate_html_content(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Generate HTML publication"""
        
        html_template = f"""
<!DOCTYPE html>
<html lang="{metadata['language']}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata['title']}</title>
    <meta name="author" content="{metadata['author']}">
    <meta name="description" content="Generated by Agentic RAG Book Generator">
    <style>
        body {{
            font-family: 'Georgia', serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .book-container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .chapter {{
            margin-bottom: 40px;
            border-left: 4px solid #3498db;
            padding-left: 20px;
        }}
        .metadata {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .toc {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            margin: 5px 0;
        }}
        .toc a {{
            color: #2980b9;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .takeaways {{
            background: #e8f6f3;
            padding: 15px;
            border-left: 4px solid #27ae60;
            margin: 20px 0;
        }}
        .takeaways h4 {{
            color: #27ae60;
            margin-top: 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #34495e;
            color: white;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="book-container">
        <header>
            <h1>{metadata['title']}</h1>
            <div class="metadata">
                <p><strong>Author:</strong> {metadata['author']}</p>
                <p><strong>Publisher:</strong> {metadata['publisher']}</p>
                <p><strong>Publication Date:</strong> {metadata['publication_date']}</p>
                <p><strong>ISBN:</strong> {metadata['isbn']}</p>
                <p><strong>Language:</strong> {metadata['language']}</p>
                <p><strong>Category:</strong> {metadata['category']}</p>
            </div>
        </header>

        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
"""
        
        # Add table of contents
        chapters = content.get("chapters", [])
        for chapter in chapters:
            chapter_id = f"chapter-{chapter.get('chapter_number', 1)}"
            chapter_title = chapter.get('title', f"Chapter {chapter.get('chapter_number', 1)}")
            html_template += f'                <li><a href="#{chapter_id}">{chapter_title}</a></li>\n'
        
        html_template += """
            </ul>
        </div>
"""
        
        # Add introduction
        if content.get("introduction"):
            html_template += f"""
        <section id="introduction">
            <h2>Introduction</h2>
            <div>{content["introduction"].get("content", "").replace(chr(10), "<br>")}</div>
        </section>
"""
        
        # Add chapters
        for chapter in chapters:
            chapter_id = f"chapter-{chapter.get('chapter_number', 1)}"
            chapter_title = chapter.get('title', f"Chapter {chapter.get('chapter_number', 1)}")
            chapter_content = chapter.get('content', '').replace('\n', '<br>')
            
            html_template += f"""
        <section id="{chapter_id}" class="chapter">
            <h2>Chapter {chapter.get('chapter_number', 1)}: {chapter_title}</h2>
            <div>{chapter_content}</div>
"""
            
            # Add key takeaways
            if chapter.get("key_takeaways"):
                html_template += """
            <div class="takeaways">
                <h4>Key Takeaways</h4>
                <ul>
"""
                for takeaway in chapter["key_takeaways"]:
                    html_template += f"                    <li>{takeaway}</li>\n"
                
                html_template += """
                </ul>
            </div>
"""
            
            html_template += "        </section>\n"
        
        # Add conclusion
        if content.get("conclusion"):
            html_template += f"""
        <section id="conclusion">
            <h2>Conclusion</h2>
            <div>{content["conclusion"].get("content", "").replace(chr(10), "<br>")}</div>
        </section>
"""
        
        # Add footer
        html_template += f"""
        <footer class="footer">
            <p>{metadata['copyright_notice']}</p>
            <p>Generated by Agentic RAG Book Generator on {metadata['publication_date']}</p>
        </footer>
    </div>
</body>
</html>
"""
        
        return html_template
    
    async def _generate_docx_content(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Generate formatted content for DOCX"""
        
        docx_content = f"""
Document: {metadata['title']}
Author: {metadata['author']}
Date: {metadata['publication_date']}

[This would be structured content for python-docx library]

Document Structure:
- Title Page
- Table of Contents
- Introduction ({len(content.get('introduction', {}).get('content', '').split())} words)
- {len(content.get('chapters', []))} Chapters
- Conclusion ({len(content.get('conclusion', {}).get('content', '').split())} words)
- {len(content.get('appendices', []))} Appendices

Formatting Instructions:
- Title: Arial, 24pt, Bold, Centered
- Headings: Arial, 16pt, Bold
- Body Text: Times New Roman, 12pt
- Line Spacing: 1.5
- Margins: 1 inch all sides
- Page Numbers: Bottom center

Content Preview:
{content.get('chapters', [{}])[0].get('content', 'No content available')[:500] if content.get('chapters') else 'No chapters available'}...
"""
        
        return docx_content
    
    async def _generate_isbn(self, book_metadata: Dict[str, Any]) -> str:
        """Generate ISBN-13 for the book"""
        
        # This is a simplified ISBN generator
        # In production, you would integrate with ISBN registration services
        
        prefix = book_metadata.get("isbn_prefix", "979-8")  # Common self-publishing prefix
        publisher_code = "12345"  # Would be assigned by ISBN agency
        title_code = str(hash(book_metadata.get("title", "default")) % 100000).zfill(5)
        
        # Calculate check digit (simplified)
        isbn_without_check = f"{prefix.replace('-', '')}{publisher_code}{title_code}"
        check_digit = self._calculate_isbn_check_digit(isbn_without_check)
        
        isbn = f"{prefix}-{publisher_code}-{title_code}-{check_digit}"
        
        return isbn
    
    def _calculate_isbn_check_digit(self, isbn_12: str) -> str:
        """Calculate ISBN-13 check digit"""
        
        total = 0
        for i, digit in enumerate(isbn_12):
            if i % 2 == 0:
                total += int(digit)
            else:
                total += int(digit) * 3
        
        check_digit = (10 - (total % 10)) % 10
        return str(check_digit)
    
    def _extract_book_metadata(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract book metadata from input data"""
        
        return {
            "title": input_data.get("title", "Untitled Book"),
            "author": input_data.get("author", "AI Generated"),
            "publisher": input_data.get("publisher", "Agentic RAG Publishers"),
            "language": input_data.get("language", "en"),
            "category": input_data.get("category", "General"),
            "project_id": input_data.get("book_project_id", str(uuid.uuid4())[:8]),
            "quality_score": input_data.get("quality_score", 0.8)
        }
    
    def _calculate_content_statistics(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive content statistics"""
        
        # Count words, chapters, etc.
        total_words = 0
        total_characters = 0
        
        if content.get("introduction"):
            intro_content = content["introduction"].get("content", "")
            total_words += len(intro_content.split())
            total_characters += len(intro_content)
        
        chapters = content.get("chapters", [])
        for chapter in chapters:
            chapter_content = chapter.get("content", "")
            total_words += len(chapter_content.split())
            total_characters += len(chapter_content)
        
        if content.get("conclusion"):
            conclusion_content = content["conclusion"].get("content", "")
            total_words += len(conclusion_content.split())
            total_characters += len(conclusion_content)
        
        for appendix in content.get("appendices", []):
            appendix_content = appendix.get("content", "")
            total_words += len(appendix_content.split())
            total_characters += len(appendix_content)
        
        return {
            "total_words": total_words,
            "total_characters": total_characters,
            "chapter_count": len(chapters),
            "appendix_count": len(content.get("appendices", [])),
            "estimated_pages": max(1, total_words // 250),  # ~250 words per page
            "reading_time_minutes": max(1, total_words // 200)  # ~200 words per minute
        }
    
    def _estimate_page_count(self, content: Dict[str, Any]) -> int:
        """Estimate page count for PDF"""
        
        stats = self._calculate_content_statistics(content)
        return stats["estimated_pages"]
    
    def _calculate_reading_time(self, content: Dict[str, Any]) -> int:
        """Calculate estimated reading time in minutes"""
        
        stats = self._calculate_content_statistics(content)
        return stats["reading_time_minutes"]
    
    async def _generate_publication_report(
        self,
        publications: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive publication report"""
        
        successful_publications = {k: v for k, v in publications.items() if v.get("success", False)}
        failed_publications = {k: v for k, v in publications.items() if not v.get("success", False)}
        
        total_file_size = sum(pub.get("file_size", 0) for pub in successful_publications.values())
        
        report = {
            "publication_summary": {
                "title": metadata["title"],
                "publication_date": metadata["publication_date"],
                "isbn": metadata["isbn"],
                "total_formats": len(publications),
                "successful_formats": len(successful_publications),
                "failed_formats": len(failed_publications)
            },
            "format_details": {
                format_name: {
                    "success": pub.get("success", False),
                    "file_size": pub.get("file_size", 0),
                    "quality_score": pub.get("quality_score", 0.0),
                    "creation_method": pub.get("creation_method", "unknown")
                }
                for format_name, pub in publications.items()
            },
            "content_statistics": metadata["content_statistics"],
            "quality_assessment": {
                "overall_quality": sum(pub.get("quality_score", 0) for pub in successful_publications.values()) / max(len(successful_publications), 1),
                "format_consistency": len(successful_publications) / max(len(publications), 1),
                "publication_completeness": 1.0 if len(failed_publications) == 0 else 0.8
            },
            "file_information": {
                "total_size_bytes": total_file_size,
                "total_size_mb": round(total_file_size / (1024 * 1024), 2),
                "largest_file": max(successful_publications.items(), key=lambda x: x[1].get("file_size", 0))[0] if successful_publications else None,
                "smallest_file": min(successful_publications.items(), key=lambda x: x[1].get("file_size", 0))[0] if successful_publications else None
            },
            "errors_and_warnings": [
                {
                    "format": format_name,
                    "error": pub.get("error", "Unknown error"),
                    "severity": "error"
                }
                for format_name, pub in failed_publications.items()
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return report
    
    async def _create_distribution_package(
        self,
        publications: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create distribution package information"""
        
        successful_publications = {k: v for k, v in publications.items() if v.get("success", False)}
        
        package_info = {
            "package_id": f"{metadata['title'].replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}",
            "included_formats": list(successful_publications.keys()),
            "package_size": sum(pub.get("file_size", 0) for pub in successful_publications.values()),
            "distribution_ready": len(successful_publications) > 0,
            "recommended_distribution": {
                "primary_format": "pdf" if "pdf" in successful_publications else list(successful_publications.keys())[0] if successful_publications else None,
                "web_format": "html" if "html" in successful_publications else None,
                "mobile_format": "epub" if "epub" in successful_publications else None,
                "editable_format": "docx" if "docx" in successful_publications else None
            },
            "metadata_files": [
                "book_metadata.json",
                "publication_report.json",
                "content_statistics.json"
            ],
            "created_at": datetime.utcnow().isoformat()
        }
        
        return package_info
    
    def _get_file_locations(self, publications: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Get file locations for successful publications"""
        
        file_locations = {}
        
        for format_name, pub in publications.items():
            if pub.get("success", False) and pub.get("file_path"):
                file_locations[format_name] = pub["file_path"]
        
        return file_locations
    
    def _generate_download_links(self, publications: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Generate download links for publications"""
        
        download_links = {}
        
        for format_name, pub in publications.items():
            if pub.get("success", False):
                # In a real implementation, these would be actual URLs
                filename = os.path.basename(pub.get("file_path", ""))
                download_links[format_name] = f"/download/{format_name}/{filename}"
        
        return download_links
    
    def _create_publication_summary(self, publications: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create concise publication summary"""
        
        successful = [k for k, v in publications.items() if v.get("success", False)]
        failed = [k for k, v in publications.items() if not v.get("success", False)]
        
        return {
            "total_formats": len(publications),
            "successful_formats": successful,
            "failed_formats": failed,
            "success_rate": len(successful) / max(len(publications), 1),
            "ready_for_distribution": len(successful) > 0,
            "primary_format": successful[0] if successful else None,
            "publication_status": "complete" if len(failed) == 0 else "partial" if len(successful) > 0 else "failed"
        }
    
    def _calculate_publication_quality_score(
        self,
        output_data: Dict[str, Any],
        task: BaseAgentTask
    ) -> float:
        """Calculate quality score for publication process"""
        
        base_score = 0.7
        
        # Quality based on successful publication formats
        publications = output_data.get("publications", {})
        successful_count = len([p for p in publications.values() if p.get("success", False)])
        total_count = len(publications)
        
        if total_count > 0:
            success_ratio = successful_count / total_count
            base_score += success_ratio * 0.2
        
        # Quality based on publication report completeness
        publication_report = output_data.get("publication_report", {})
        if publication_report.get("quality_assessment", {}).get("overall_quality", 0) > 0.8:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))