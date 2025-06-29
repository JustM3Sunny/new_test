#!/usr/bin/env python3

"""
Web Search and RAG (Retrieval Augmented Generation) Module for CODY Agent
Provides real-time web search, documentation retrieval, and context integration
"""

import re
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin
import logging

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

logger = logging.getLogger('CODY.WebSearch')

@dataclass
class SearchResult:
    """Represents a search result."""
    title: str
    url: str
    snippet: str
    relevance_score: float
    source_type: str
    metadata: Dict[str, Any]

class WebSearchRAG:
    """Web search and RAG system for programming assistance."""
    
    def __init__(self):
        self.search_engines = {
           'google': 'https://www.google.com/search?q={}',
'stackoverflow': 'https://stackoverflow.com/search?q={}',
'github': 'https://github.com/search?q={}',
'gitlab': 'https://gitlab.com/search?search={}',
'replit': 'https://replit.com/search?q={}',
'devto': 'https://dev.to/search?q={}',
'medium': 'https://medium.com/search?q={}',
'hackernoon': 'https://hackernoon.com/search?query={}',
'reddit': 'https://www.reddit.com/search/?q={}',
'duckduckgo': 'https://duckduckgo.com/html/?q={}',
'bing': 'https://www.bing.com/search?q={}',
'yahoo': 'https://search.yahoo.com/search?p={}',
'ask': 'https://www.ask.com/web?q={}',
'yandex': 'https://yandex.ru/search/?text={}',
'baidu': 'https://www.baidu.com/s?wd={}',
'naver': 'https://search.naver.com/search.naver?query={}',
'daum': 'https://search.daum.net/search?q={}',
'nate': 'https://search.nate.com/search?query={}',
'google': 'https://www.google.com/search?q={}',
'bing': 'https://www.bing.com/search?q={}',
'yahoo': 'https://search.yahoo.com/search?p={}',
'ask': 'https://www.ask.com/web?q={}',
'yandex': 'https://yandex.ru/search/?text={}',
'baidu': 'https://www.baidu.com/s?wd={}',
'naver': 'https://search.naver.com/search.naver?query={}',
'daum': 'https://search.daum.net/search?q={}',
'nate': 'https://search.nate.com/search?query={}',
        }
        
        self.programming_sites = {
            'stackoverflow.com': {'weight': 1.0, 'type': 'qa'},
            'github.com': {'weight': 0.9, 'type': 'code'},
            'docs.python.org': {'weight': 0.95, 'type': 'docs'},
            'developer.mozilla.org': {'weight': 0.95, 'type': 'docs'},
            'w3schools.com': {'weight': 0.8, 'type': 'tutorial'},
            'geeksforgeeks.org': {'weight': 0.7, 'type': 'tutorial'},
            'medium.com': {'weight': 0.6, 'type': 'article'},
            'dev.to': {'weight': 0.7, 'type': 'article'},
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def search_programming_help(self, query: str, max_results: int = 5, search_type: str = 'general') -> List[SearchResult]:
        """
        Search for programming help and documentation.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_type: Type of search ('general', 'stackoverflow', 'github', 'docs')
            
        Returns:
            List of SearchResult objects
        """
        if not WEB_SEARCH_AVAILABLE:
            logger.warning("Web search not available - missing dependencies")
            return []
        
        # Enhance query for programming context
        enhanced_query = self._enhance_programming_query(query)
        
        results = []
        
        if search_type == 'stackoverflow':
            results.extend(self._search_stackoverflow(enhanced_query, max_results))
        elif search_type == 'github':
            results.extend(self._search_github(enhanced_query, max_results))
        elif search_type == 'docs':
            results.extend(self._search_documentation(enhanced_query, max_results))
        else:
            # General search across multiple sources
            results.extend(self._search_general(enhanced_query, max_results))
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]
    
    def _enhance_programming_query(self, query: str) -> str:
        """Enhance query with programming-specific terms."""
        # Add programming context if not present
        programming_indicators = ['error', 'function', 'class', 'method', 'variable', 'syntax', 'code', 'programming']
        
        if not any(indicator in query.lower() for indicator in programming_indicators):
            # Try to detect programming language
            languages = [
                'python', 'javascript', 'java', 'cpp', 'c++', 'go', 'rust',
                'html', 'css', 'typescript', 'react', 'nodejs', 'express',
                'flask', 'django', 'sql', 'mysql', 'postgresql', 'mongodb',
                'redis', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
                'firebase', 'react-native', 'flutter', 'swift', 'kotlin', 'dart',
                'ruby', 'rails', 'php', 'laravel', 'symfony', 'vue',
                'angular', 'svelte', 'nextjs', 'nuxtjs', 'remix',
                'unity', 'unreal-engine', 'android', 'ios', 'deno', 'scala', 'perl', 'haskell', 'elixir', 'clojure', 'f#'
            ]
            
            detected_lang = None
            
            for lang in languages:
                if lang in query.lower():
                    detected_lang = lang
                    break
            
            if detected_lang:
                query = f"{query} {detected_lang} programming"
            else:
                query = f"{query} programming code"
        
        return query
    
    def _search_general(self, query: str, max_results: int) -> List[SearchResult]:
        """Perform general web search."""
        results = []
        
        try:
            # Use DuckDuckGo as it doesn't require API keys
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse DuckDuckGo results
            for result_div in soup.find_all('div', class_='result')[:max_results * 2]:
                title_elem = result_div.find('a', class_='result__a')
                snippet_elem = result_div.find('a', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True)
                    
                    # Calculate relevance score
                    relevance = self._calculate_relevance(title, snippet, url, query)
                    
                    if relevance > 0.3:  # Minimum relevance threshold
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            relevance_score=relevance,
                            source_type='general',
                            metadata={'search_engine': 'duckduckgo'}
                        ))
        
        except Exception as e:
            logger.error(f"General search failed: {e}")
        
        return results
    
    def _search_stackoverflow(self, query: str, max_results: int) -> List[SearchResult]:
        """Search Stack Overflow specifically."""
        results = []
        
        try:
            # Use Stack Overflow API
            api_url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'q': query,
                'site': 'stackoverflow',
                'pagesize': max_results,
                'filter': 'withbody'
            }
            
            response = self.session.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get('items', []):
                title = item.get('title', '')
                url = item.get('link', '')
                snippet = BeautifulSoup(item.get('body', ''), 'html.parser').get_text()[:300]
                
                # Stack Overflow results are highly relevant for programming
                relevance = 0.9 + (item.get('score', 0) / 100.0)
                relevance = min(1.0, relevance)
                
                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    relevance_score=relevance,
                    source_type='qa',
                    metadata={
                        'score': item.get('score', 0),
                        'answer_count': item.get('answer_count', 0),
                        'tags': item.get('tags', [])
                    }
                ))
        
        except Exception as e:
            logger.error(f"Stack Overflow search failed: {e}")
        
        return results
    
    def _search_github(self, query: str, max_results: int) -> List[SearchResult]:
        """Search GitHub for code examples."""
        results = []
        
        try:
            # Use GitHub API (no auth required for search)
            api_url = "https://api.github.com/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': max_results
            }
            
            response = self.session.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get('items', []):
                title = item.get('full_name', '')
                url = item.get('html_url', '')
                snippet = item.get('description', '') or 'No description available'
                
                # Calculate relevance based on stars and relevance
                stars = item.get('stargazers_count', 0)
                relevance = 0.7 + min(0.3, stars / 1000.0)
                
                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    relevance_score=relevance,
                    source_type='code',
                    metadata={
                        'stars': stars,
                        'language': item.get('language', ''),
                        'updated_at': item.get('updated_at', '')
                    }
                ))
        
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
        
        return results
    
    def _search_documentation(self, query: str, max_results: int) -> List[SearchResult]:
        """Search official documentation sites."""
        results = []
        
        # Search Python documentation
        if 'python' in query.lower():
            results.extend(self._search_python_docs(query, max_results // 2))
        
        # Search MDN for JavaScript/Web
        if any(term in query.lower() for term in ['javascript', 'js', 'html', 'css', 'web']):
            results.extend(self._search_mdn_docs(query, max_results // 2))
        
        return results
    
    def _search_python_docs(self, query: str, max_results: int) -> List[SearchResult]:
        """Search Python official documentation."""
        results = []
        
        try:
            search_url = f"https://docs.python.org/3/search.html?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for result in soup.find_all('li', class_='search-result')[:max_results]:
                title_elem = result.find('a')
                snippet_elem = result.find('p')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = urljoin('https://docs.python.org/3/', title_elem.get('href', ''))
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        relevance_score=0.95,  # Official docs are highly relevant
                        source_type='docs',
                        metadata={'source': 'python_docs'}
                    ))
        
        except Exception as e:
            logger.error(f"Python docs search failed: {e}")
        
        return results
    
    def _search_mdn_docs(self, query: str, max_results: int) -> List[SearchResult]:
        """Search MDN Web Docs."""
        results = []
        
        try:
            search_url = f"https://developer.mozilla.org/en-US/search?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for result in soup.find_all('div', class_='result-item')[:max_results]:
                title_elem = result.find('h4').find('a') if result.find('h4') else None
                snippet_elem = result.find('p')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = urljoin('https://developer.mozilla.org', title_elem.get('href', ''))
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        relevance_score=0.95,  # Official docs are highly relevant
                        source_type='docs',
                        metadata={'source': 'mdn'}
                    ))
        
        except Exception as e:
            logger.error(f"MDN search failed: {e}")
        
        return results
    
    def _calculate_relevance(self, title: str, snippet: str, url: str, query: str) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        query_terms = query.lower().split()
        
        # Title relevance (weighted heavily)
        title_lower = title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        score += (title_matches / len(query_terms)) * 0.5
        
        # Snippet relevance
        snippet_lower = snippet.lower()
        snippet_matches = sum(1 for term in query_terms if term in snippet_lower)
        score += (snippet_matches / len(query_terms)) * 0.3
        
        # URL/domain relevance
        domain = self._extract_domain(url)
        if domain in self.programming_sites:
            site_info = self.programming_sites[domain]
            score += site_info['weight'] * 0.2
        
        return min(1.0, score)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except:
            return ''
    
    def get_code_examples(self, query: str, language: str = None) -> List[Dict[str, Any]]:
        """Get code examples for a specific query."""
        search_query = query
        if language:
            search_query = f"{query} {language} example code"
        
        results = self.search_programming_help(search_query, max_results=10, search_type='github')
        
        code_examples = []
        for result in results:
            if result.source_type == 'code':
                code_examples.append({
                    'title': result.title,
                    'url': result.url,
                    'description': result.snippet,
                    'language': result.metadata.get('language', ''),
                    'stars': result.metadata.get('stars', 0)
                })
        
        return code_examples
    
    def get_error_solutions(self, error_message: str, language: str = None) -> List[SearchResult]:
        """Get solutions for specific error messages."""
        # Clean up error message for better search
        cleaned_error = re.sub(r'File ".*?"', '', error_message)  # Remove file paths
        cleaned_error = re.sub(r'line \d+', '', cleaned_error)    # Remove line numbers
        cleaned_error = cleaned_error.strip()
        
        search_query = f"{cleaned_error} solution fix"
        if language:
            search_query = f"{search_query} {language}"
        
        return self.search_programming_help(search_query, max_results=5, search_type='stackoverflow')
    
    def summarize_search_results(self, results: List[SearchResult]) -> str:
        """Summarize search results into a coherent response."""
        if not results:
            return "No relevant results found."
        
        summary_parts = []
        
        # Group results by type
        docs_results = [r for r in results if r.source_type == 'docs']
        qa_results = [r for r in results if r.source_type == 'qa']
        code_results = [r for r in results if r.source_type == 'code']
        
        if docs_results:
            summary_parts.append("**Official Documentation:**")
            for result in docs_results[:2]:
                summary_parts.append(f"- {result.title}: {result.snippet[:100]}...")
        
        if qa_results:
            summary_parts.append("\n**Community Q&A:**")
            for result in qa_results[:2]:
                summary_parts.append(f"- {result.title}: {result.snippet[:100]}...")
        
        if code_results:
            summary_parts.append("\n**Code Examples:**")
            for result in code_results[:2]:
                summary_parts.append(f"- {result.title}: {result.snippet[:100]}...")
        
        return "\n".join(summary_parts)
