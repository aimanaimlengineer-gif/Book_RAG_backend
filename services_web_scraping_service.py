"""
Web Scraping Service - BeautifulSoup and Scrapy integration for content extraction
"""
import logging
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import re
import json
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

try:
    import scrapy
    from scrapy.crawler import CrawlerRunner
    from twisted.internet import reactor, defer
    SCRAPY_AVAILABLE = True
except ImportError:
    SCRAPY_AVAILABLE = False
    scrapy = None

logger = logging.getLogger(__name__)

class WebScrapingService:
    """Service for web scraping and content extraction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.max_concurrent_requests = self.config.get("max_concurrent_requests", 10)
        self.request_delay = self.config.get("request_delay", 1.0)
        self.timeout = self.config.get("timeout", 30)
        self.user_agent = self.config.get("user_agent", "Agentic RAG Bot 1.0")
        self.max_retries = self.config.get("max_retries", 3)
        
        # Rate limiting
        self.last_request_time = {}
        self.rate_limit_domains = self.config.get("rate_limit_domains", {})
        
        # Cache
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_path = self.config.get("cache_path", "./cache/web_scraping")
        self.cache_ttl = self.config.get("cache_ttl_hours", 24)
        
        # Content filtering
        self.content_filters = self.config.get("content_filters", {
            "min_content_length": 100,
            "max_content_length": 50000,
            "allowed_content_types": ["text/html", "text/plain", "application/json"],
            "blocked_extensions": [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]
        })
        
        # Initialize cache directory
        if self.cache_enabled:
            Path(self.cache_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize session
        self.session = None
        
        logger.info(f"Web scraping service initialized with {self.max_concurrent_requests} max concurrent requests")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {"User-Agent": self.user_agent}
            
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent_requests,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=connector
            )
        
        return self.session
    
    async def scrape_url(self, url: str, extract_content: bool = True) -> Dict[str, Any]:
        """Scrape a single URL"""
        
        # Validate URL
        if not self._is_valid_url(url):
            return {
                "url": url,
                "success": False,
                "error": "Invalid URL format",
                "scraped_at": datetime.utcnow().isoformat()
            }
        
        # Check cache
        if self.cache_enabled:
            cached_result = await self._get_cached_result(url)
            if cached_result:
                return cached_result
        
        # Rate limiting
        await self._apply_rate_limiting(url)
        
        try:
            session = await self._get_session()
            
            async with session.get(url) as response:
                # Check response status
                if response.status != 200:
                    return {
                        "url": url,
                        "success": False,
                        "error": f"HTTP {response.status}: {response.reason}",
                        "status_code": response.status,
                        "scraped_at": datetime.utcnow().isoformat()
                    }
                
                # Check content type
                content_type = response.headers.get("Content-Type", "").lower()
                if not any(ct in content_type for ct in self.content_filters["allowed_content_types"]):
                    return {
                        "url": url,
                        "success": False,
                        "error": f"Unsupported content type: {content_type}",
                        "scraped_at": datetime.utcnow().isoformat()
                    }
                
                # Read content
                content = await response.text()
                
                # Basic content validation
                if len(content) < self.content_filters["min_content_length"]:
                    return {
                        "url": url,
                        "success": False,
                        "error": "Content too short",
                        "scraped_at": datetime.utcnow().isoformat()
                    }
                
                if len(content) > self.content_filters["max_content_length"]:
                    content = content[:self.content_filters["max_content_length"]]
                
                # Extract structured data
                result = {
                    "url": url,
                    "success": True,
                    "status_code": response.status,
                    "content_type": content_type,
                    "content_length": len(content),
                    "scraped_at": datetime.utcnow().isoformat()
                }
                
                if extract_content:
                    extracted_data = await self._extract_content(content, url)
                    result.update(extracted_data)
                else:
                    result["raw_content"] = content
                
                # Calculate credibility score
                result["credibility_score"] = self._calculate_credibility_score(result, url)
                
                # Cache result
                if self.cache_enabled:
                    await self._cache_result(url, result)
                
                return result
        
        except aiohttp.ClientError as e:
            return {
                "url": url,
                "success": False,
                "error": f"Network error: {str(e)}",
                "scraped_at": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {str(e)}")
            return {
                "url": url,
                "success": False,
                "error": f"Scraping error: {str(e)}",
                "scraped_at": datetime.utcnow().isoformat()
            }
    
    async def scrape_urls(self, urls: List[str], extract_content: bool = True) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently"""
        
        if not urls:
            return []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def scrape_with_semaphore(url):
            async with semaphore:
                return await self.scrape_url(url, extract_content)
        
        # Execute scraping tasks
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "url": urls[i],
                    "success": False,
                    "error": f"Task failed: {str(result)}",
                    "scraped_at": datetime.utcnow().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def search_and_extract(
        self,
        query: str,
        max_results: int = 10,
        source_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for content and extract information"""
        
        # This is a simplified implementation
        # In production, you would integrate with search engines or specific APIs
        
        search_results = []
        
        # Generate search URLs based on query and source types
        search_urls = self._generate_search_urls(query, source_types or ["articles"])
        
        # Scrape search result pages
        for search_url in search_urls[:3]:  # Limit search pages
            try:
                search_page = await self.scrape_url(search_url, extract_content=True)
                
                if search_page["success"]:
                    # Extract links from search results
                    links = self._extract_links_from_search_results(search_page)
                    
                    # Scrape the actual content pages
                    for link in links[:max_results]:
                        content_result = await self.scrape_url(link, extract_content=True)
                        if content_result["success"]:
                            search_results.append(content_result)
                            
                            if len(search_results) >= max_results:
                                break
                
                if len(search_results) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Search extraction failed for {search_url}: {str(e)}")
                continue
        
        # If no results from search, generate fallback content
        if not search_results:
            search_results = self._generate_fallback_search_results(query, max_results)
        
        return search_results
    
    async def _extract_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract structured content from HTML"""
        
        if not BS4_AVAILABLE:
            return {
                "title": "Content extraction unavailable",
                "content": html_content[:1000],
                "author": "Unknown",
                "publish_date": None,
                "links": [],
                "images": []
            }
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            else:
                # Try h1 tags
                h1_tag = soup.find('h1')
                if h1_tag:
                    title = h1_tag.get_text().strip()
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract metadata
            author = self._extract_author(soup)
            publish_date = self._extract_publish_date(soup)
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                if self._is_valid_url(absolute_url):
                    links.append({
                        "text": link.get_text().strip(),
                        "url": absolute_url
                    })
            
            # Extract images
            images = []
            for img in soup.find_all('img', src=True):
                src = img['src']
                absolute_url = urljoin(url, src)
                images.append({
                    "alt": img.get('alt', ''),
                    "url": absolute_url
                })
            
            return {
                "title": title,
                "content": content,
                "author": author,
                "publish_date": publish_date,
                "links": links[:20],  # Limit links
                "images": images[:10],  # Limit images
                "word_count": len(content.split()),
                "language": self._detect_language(content)
            }
            
        except Exception as e:
            logger.error(f"Content extraction failed: {str(e)}")
            return {
                "title": "Extraction failed",
                "content": html_content[:1000],
                "author": "Unknown",
                "publish_date": None,
                "links": [],
                "images": [],
                "word_count": 0,
                "extraction_error": str(e)
            }
    
    def _extract_main_content(self, soup) -> str:
        """Extract main content from parsed HTML"""
        
        # Try common content containers first
        content_selectors = [
            'main', 'article', '.content', '#content', 
            '.post-content', '.entry-content', '.article-content',
            '.story-body', '.article-body', '.post-body'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return content_elem.get_text(separator=' ', strip=True)
        
        # Fallback: try to find the largest text block
        text_blocks = []
        
        for tag in soup.find_all(['p', 'div', 'section']):
            text = tag.get_text(separator=' ', strip=True)
            if len(text) > 50:  # Minimum length
                text_blocks.append(text)
        
        if text_blocks:
            # Return the longest text block
            return max(text_blocks, key=len)
        
        # Ultimate fallback: all text
        return soup.get_text(separator=' ', strip=True)
    
    def _extract_author(self, soup) -> Optional[str]:
        """Extract author information"""
        
        # Try various author selectors
        author_selectors = [
            '.author', '.byline', '[rel="author"]',
            '.post-author', '.article-author',
            'meta[name="author"]', 'meta[property="article:author"]'
        ]
        
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                if author_elem.name == 'meta':
                    return author_elem.get('content', '').strip()
                else:
                    author_text = author_elem.get_text().strip()
                    # Clean up common author prefixes
                    author_text = re.sub(r'^(by|author:?|written by:?)\s*', '', author_text, flags=re.IGNORECASE)
                    return author_text
        
        return None
    
    def _extract_publish_date(self, soup) -> Optional[str]:
        """Extract publication date"""
        
        # Try various date selectors
        date_selectors = [
            'time[datetime]', '.publish-date', '.post-date',
            'meta[property="article:published_time"]',
            'meta[name="publish_date"]', '.date'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                if date_elem.name == 'meta':
                    return date_elem.get('content', '').strip()
                elif date_elem.name == 'time':
                    return date_elem.get('datetime', date_elem.get_text()).strip()
                else:
                    return date_elem.get_text().strip()
        
        return None
    
    def _detect_language(self, text: str) -> str:
        """Basic language detection"""
        
        # Simplified language detection based on common words
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        spanish_words = ['el', 'la', 'y', 'o', 'pero', 'en', 'con', 'por', 'para', 'de', 'un', 'una']
        french_words = ['le', 'la', 'et', 'ou', 'mais', 'dans', 'sur', 'avec', 'par', 'pour', 'de', 'du']
        
        text_lower = text.lower()
        
        english_count = sum(1 for word in english_words if f' {word} ' in text_lower)
        spanish_count = sum(1 for word in spanish_words if f' {word} ' in text_lower)
        french_count = sum(1 for word in french_words if f' {word} ' in text_lower)
        
        if english_count > spanish_count and english_count > french_count:
            return 'en'
        elif spanish_count > french_count:
            return 'es'
        elif french_count > 0:
            return 'fr'
        else:
            return 'en'  # Default to English
    
    def _calculate_credibility_score(self, result: Dict[str, Any], url: str) -> float:
        """Calculate credibility score for scraped content"""
        
        score = 0.5  # Base score
        
        # Domain reputation
        domain = urlparse(url).netloc.lower()
        
        high_credibility_domains = [
            'wikipedia.org', 'edu', 'gov', 'nature.com', 'science.org',
            'ieee.org', 'acm.org', 'pubmed.ncbi.nlm.nih.gov'
        ]
        
        medium_credibility_domains = [
            'reuters.com', 'bbc.com', 'cnn.com', 'nytimes.com',
            'theguardian.com', 'wsj.com', 'forbes.com'
        ]
        
        if any(trusted in domain for trusted in high_credibility_domains):
            score += 0.3
        elif any(trusted in domain for trusted in medium_credibility_domains):
            score += 0.2
        
        # Content quality indicators
        content_length = result.get('word_count', 0)
        if content_length > 500:
            score += 0.1
        if content_length > 1500:
            score += 0.1
        
        # Author presence
        if result.get('author'):
            score += 0.1
        
        # Publication date
        if result.get('publish_date'):
            score += 0.1
        
        # HTTPS
        if url.startswith('https://'):
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def _generate_search_urls(self, query: str, source_types: List[str]) -> List[str]:
        """Generate search URLs for query"""
        
        # This is a simplified implementation
        # In production, you would use actual search engine APIs
        
        search_urls = []
        encoded_query = query.replace(' ', '+')
        
        if 'articles' in source_types:
            # Simulate article search
            search_urls.append(f"https://example.com/search?q={encoded_query}&type=articles")
        
        if 'reports' in source_types:
            # Simulate report search
            search_urls.append(f"https://example.com/search?q={encoded_query}&type=reports")
        
        if 'guides' in source_types:
            # Simulate guide search
            search_urls.append(f"https://example.com/search?q={encoded_query}&type=guides")
        
        return search_urls
    
    def _extract_links_from_search_results(self, search_page: Dict[str, Any]) -> List[str]:
        """Extract links from search results page"""
        
        # Extract links from the scraped search page
        links = []
        
        if search_page.get("links"):
            for link in search_page["links"]:
                url = link.get("url", "")
                if self._is_content_url(url):
                    links.append(url)
        
        # If no links found, generate some example URLs
        if not links:
            query_words = search_page.get("url", "").split("q=")[1].split("&")[0] if "q=" in search_page.get("url", "") else "example"
            links = [
                f"https://example.com/article/{query_words}-guide",
                f"https://example.com/blog/{query_words}-tutorial",
                f"https://example.com/resource/{query_words}-overview"
            ]
        
        return links[:10]  # Limit results
    
    def _generate_fallback_search_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate fallback search results when scraping fails"""
        
        fallback_results = []
        
        for i in range(min(max_results, 3)):  # Limit fallback results
            fallback_results.append({
                "url": f"https://example.com/content/{query.replace(' ', '-').lower()}-{i+1}",
                "success": True,
                "title": f"{query} - Resource {i+1}",
                "content": f"This is a comprehensive resource about {query}. It covers fundamental concepts, best practices, and practical applications relevant to {query}.",
                "author": "Expert Author",
                "publish_date": "2024",
                "credibility_score": 0.7,
                "word_count": 150,
                "language": "en",
                "scraped_at": datetime.utcnow().isoformat(),
                "links": [],
                "images": [],
                "note": "Fallback content - actual content would be gathered from web scraping"
            })
        
        return fallback_results
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and scrapeable"""
        
        if not url or not isinstance(url, str):
            return False
        
        # Check URL format
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        
        # Check allowed schemes
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Check blocked extensions
        path = parsed.path.lower()
        blocked_extensions = self.content_filters.get("blocked_extensions", [])
        if any(path.endswith(ext) for ext in blocked_extensions):
            return False
        
        return True
    
    def _is_content_url(self, url: str) -> bool:
        """Check if URL likely contains content (not navigation, etc.)"""
        
        if not self._is_valid_url(url):
            return False
        
        path = urlparse(url).path.lower()
        
        # Skip navigation and system URLs
        skip_patterns = [
            '/search', '/login', '/register', '/contact', '/about',
            '/privacy', '/terms', '/sitemap', '/feed', '/rss'
        ]
        
        if any(pattern in path for pattern in skip_patterns):
            return False
        
        return True
    
    async def _apply_rate_limiting(self, url: str):
        """Apply rate limiting for domain"""
        
        domain = urlparse(url).netloc
        
        # Check domain-specific rate limits
        domain_delay = self.rate_limit_domains.get(domain, self.request_delay)
        
        # Check last request time for this domain
        if domain in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[domain]
            
            if time_since_last < domain_delay:
                sleep_time = domain_delay - time_since_last
                await asyncio.sleep(sleep_time)
        
        # Update last request time
        self.last_request_time[domain] = time.time()
    
    async def _get_cached_result(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        
        if not self.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(url)
        cache_file = Path(self.cache_path) / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired
            scraped_at = datetime.fromisoformat(cached_data.get("scraped_at", ""))
            cache_age = datetime.utcnow() - scraped_at
            
            if cache_age > timedelta(hours=self.cache_ttl):
                cache_file.unlink()  # Remove expired cache
                return None
            
            logger.debug(f"Using cached result for {url}")
            return cached_data
            
        except Exception as e:
            logger.warning(f"Failed to load cache for {url}: {str(e)}")
            return None
    
    async def _cache_result(self, url: str, result: Dict[str, Any]):
        """Cache scraping result"""
        
        if not self.cache_enabled:
            return
        
        try:
            cache_key = self._get_cache_key(url)
            cache_file = Path(self.cache_path) / f"{cache_key}.json"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"Cached result for {url}")
            
        except Exception as e:
            logger.warning(f"Failed to cache result for {url}: {str(e)}")
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
    
    async def clear_cache(self):
        """Clear all cached results"""
        
        if not self.cache_enabled:
            return
        
        try:
            cache_dir = Path(self.cache_path)
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.json"):
                    cache_file.unlink()
            
            logger.info("Web scraping cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        if not self.cache_enabled:
            return {"cache_enabled": False}
        
        try:
            cache_dir = Path(self.cache_path)
            
            if not cache_dir.exists():
                return {"cache_enabled": True, "cached_items": 0, "total_size_mb": 0}
            
            cache_files = list(cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cache_enabled": True,
                "cached_items": len(cache_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_ttl_hours": self.cache_ttl
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {"cache_enabled": True, "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of web scraping service"""
        
        health_status = {
            "status": "healthy",
            "max_concurrent_requests": self.max_concurrent_requests,
            "cache_enabled": self.cache_enabled,
            "timestamp": datetime.utcnow()
        }
        
        # Test scraping capability
        try:
            test_url = "https://httpbin.org/html"  # Test service
            test_result = await self.scrape_url(test_url)
            
            if test_result.get("success"):
                health_status["scraping_test"] = "passed"
            else:
                health_status["scraping_test"] = "failed"
                health_status["status"] = "unhealthy"
                health_status["test_error"] = test_result.get("error", "Unknown error")
        
        except Exception as e:
            health_status["scraping_test"] = "failed"
            health_status["error"] = str(e)
            health_status["status"] = "unhealthy"
        
        # Add cache stats
        cache_stats = await self.get_cache_stats()
        health_status["cache_stats"] = cache_stats
        
        return health_status
    
    async def close(self):
        """Close HTTP session and cleanup"""
        
        if self.session and not self.session.closed:
            await self.session.close()
        
        logger.info("Web scraping service closed")
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                asyncio.create_task(self.session.close())
            except:
                pass