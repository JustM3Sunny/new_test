#!/usr/bin/env python3

"""
Natural Language Processing Module for CODY Agent
Handles intent recognition, text-to-code conversion, and context understanding
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger('CODY.NLP')

class Intent(Enum):
    """User intent categories."""
    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    READ_FILE = "read_file"
    DELETE_FILE = "delete_file"
    SEARCH_CODE = "search_code"
    DEBUG_CODE = "debug_code"
    REFACTOR_CODE = "refactor_code"
    GENERATE_TESTS = "generate_tests"
    RUN_COMMAND = "run_command"
    EXPLAIN_CODE = "explain_code"
    CONVERT_CODE = "convert_code"
    HELP = "help"
    UNKNOWN = "unknown"

@dataclass
class IntentResult:
    """Result of intent recognition."""
    intent: Intent
    confidence: float
    entities: Dict[str, Any]
    raw_text: str
    processed_text: str

class NLPProcessor:
    """Advanced NLP processor for understanding user commands."""
    
    def __init__(self):
        self.intent_patterns = self._initialize_patterns()
        self.language_keywords = self._initialize_language_keywords()
        
    def _initialize_patterns(self) -> Dict[Intent, List[str]]:
        """Initialize regex patterns for intent recognition."""
        return {
            Intent.CREATE_FILE: [
                r"(?:create|make|generate|build)\s+(?:a\s+)?(?:new\s+)?(?:file|script|program|class|function)",
                r"(?:mujhe|main)\s+(?:chahiye|banana|create)\s+(?:file|script)",
                r"(?:i\s+)?(?:need|want)\s+(?:a\s+)?(?:new\s+)?(?:file|script|program)",
                r"(?:write|code)\s+(?:a\s+)?(?:new\s+)?(?:file|script|program|function)",
            ],
            Intent.EDIT_FILE: [
                r"(?:edit|modify|change|update|fix)\s+(?:the\s+)?(?:file|code|function|class)",
                r"(?:replace|substitute)\s+(?:this|that)\s+(?:with|by)",
                r"(?:add|insert)\s+(?:this|that|code|function)\s+(?:to|in|into)",
                r"(?:remove|delete)\s+(?:this|that|line|function|code)",
            ],
            Intent.READ_FILE: [
                r"(?:read|show|display|open|view)\s+(?:the\s+)?(?:file|code|content)",
                r"(?:what|how)\s+(?:is|does)\s+(?:in|inside)\s+(?:this|that|the)\s+file",
                r"(?:show|tell)\s+me\s+(?:the\s+)?(?:content|code)",
            ],
            Intent.SEARCH_CODE: [
                r"(?:search|find|look\s+for|locate)\s+(?:for\s+)?(?:function|class|variable|code)",
                r"(?:where\s+is|find\s+me)\s+(?:the\s+)?(?:function|class|variable)",
                r"(?:grep|search)\s+(?:for\s+)?['\"].*['\"]",
            ],
            Intent.DEBUG_CODE: [
                r"(?:debug|fix|solve|resolve)\s+(?:this|that|the)\s+(?:error|bug|issue|problem)",
                r"(?:why\s+is|what's\s+wrong)\s+(?:with\s+)?(?:this|that|the)\s+code",
                r"(?:error|exception|bug)\s+(?:in|with|at)",
            ],
            Intent.REFACTOR_CODE: [
                r"(?:refactor|improve|optimize|clean\s+up)\s+(?:this|that|the)\s+code",
                r"(?:make\s+this|make\s+that)\s+(?:better|cleaner|more\s+efficient)",
                r"(?:extract|separate)\s+(?:function|method|class)",
            ],
            Intent.GENERATE_TESTS: [
                r"(?:generate|create|write)\s+(?:tests|test\s+cases|unit\s+tests)",
                r"(?:test|testing)\s+(?:this|that|the)\s+(?:function|class|code)",
                r"(?:add|write)\s+(?:test\s+coverage|tests)\s+(?:for|to)",
            ],
            Intent.RUN_COMMAND: [
                r"(?:run|execute|start)\s+(?:the\s+)?(?:command|script|program)",
                r"(?:install|npm|pip|yarn)\s+",
                r"(?:git\s+|docker\s+|python\s+|node\s+)",
                r"(?:show|get|display)\s+(?:me\s+)?(?:current\s+)?(?:time|date)",
                r"(?:what\s+)?(?:time|date)\s+(?:is\s+it|now)",
                r"(?:ls|dir|list)\s+(?:files|directories)",
                r"(?:pwd|current\s+directory|where\s+am\s+i)",
                r"(?:remove|delete|rm)\s+(?:files|directories|pychache|__pycache__)",
                r"(?:clean|cleanup)\s+(?:cache|temp|temporary)",
            ],
            Intent.EXPLAIN_CODE: [
                r"(?:explain|describe|tell\s+me\s+about)\s+(?:this|that|the)\s+code",
                r"(?:what\s+does|how\s+does)\s+(?:this|that)\s+(?:do|work|function)",
                r"(?:help\s+me\s+understand|explain\s+to\s+me)",
            ],
            Intent.CONVERT_CODE: [
                r"(?:convert|translate|transform)\s+(?:this|that)\s+(?:to|into|from)",
                r"(?:change|rewrite)\s+(?:this|that)\s+(?:from|to)\s+(?:python|javascript|java)",
                r"(?:port|migrate)\s+(?:this|that)\s+(?:to|from)",
            ],
        }
    
    def _initialize_language_keywords(self) -> Dict[str, List[str]]:
        """Initialize programming language keywords for detection."""
        return {
            'python': ['def', 'class', 'import', 'from', 'if', 'elif', 'else', 'for', 'while', 'try', 'except'],
            'javascript': ['function', 'const', 'let', 'var', 'if', 'else', 'for', 'while', 'try', 'catch'],
            'java': ['public', 'private', 'class', 'interface', 'extends', 'implements', 'if', 'else', 'for', 'while'],
            'cpp': ['#include', 'class', 'struct', 'namespace', 'if', 'else', 'for', 'while', 'try', 'catch'],
            'go': ['func', 'package', 'import', 'if', 'else', 'for', 'range', 'switch', 'case'],
            'rust': ['fn', 'struct', 'enum', 'impl', 'if', 'else', 'for', 'while', 'match', 'let'],
        }
    
    def process_natural_language(self, text: str) -> IntentResult:
        """
        Process natural language input and extract intent and entities.
        
        Args:
            text: User input text
            
        Returns:
            IntentResult with detected intent and extracted entities
        """
        text_lower = text.lower().strip()
        
        # Detect intent
        intent, confidence = self._detect_intent(text_lower)
        
        # Extract entities
        entities = self._extract_entities(text, intent)
        
        # Process text for better understanding
        processed_text = self._preprocess_text(text)
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            entities=entities,
            raw_text=text,
            processed_text=processed_text
        )
    
    def _detect_intent(self, text: str) -> Tuple[Intent, float]:
        """Detect user intent from text with improved sensitivity."""
        best_intent = Intent.UNKNOWN
        best_confidence = 0.0

        # Normalize text for better matching
        normalized_text = text.lower().strip()

        # Quick checks for common patterns first
        quick_patterns = {
            Intent.RUN_COMMAND: [
                'time', 'date', 'current time', 'show me time', 'what time',
                'ls files', 'list files', 'dir', 'pwd', 'current directory',
                'remove', 'delete', 'rm', 'pychache', '__pycache__'
            ]
        }

        # Check quick patterns first for better performance
        for intent, quick_words in quick_patterns.items():
            for word in quick_words:
                if word in normalized_text:
                    # Calculate confidence based on how much of the text matches
                    confidence = len(word) / len(normalized_text)
                    if confidence > best_confidence:
                        best_confidence = min(0.8, confidence + 0.2)  # Boost quick matches
                        best_intent = intent

        # Then check regex patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                try:
                    match = re.search(pattern, normalized_text, re.IGNORECASE | re.UNICODE)
                    if match:
                        # Improved confidence calculation
                        match_length = len(match.group())
                        text_length = len(normalized_text)

                        # Base confidence on match coverage
                        coverage = match_length / text_length

                        # Bonus for exact matches or high coverage
                        if coverage > 0.7:
                            confidence = min(0.95, coverage + 0.2)
                        elif coverage > 0.4:
                            confidence = min(0.8, coverage + 0.1)
                        else:
                            confidence = coverage

                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_intent = intent
                except re.error:
                    # Skip invalid regex patterns
                    continue

        # Boost confidence for exact command matches
        if normalized_text.startswith('/'):
            best_confidence = min(1.0, best_confidence + 0.3)

        # Ensure minimum confidence for recognized patterns
        if best_intent != Intent.UNKNOWN and best_confidence < 0.3:
            best_confidence = 0.3

        return best_intent, min(1.0, best_confidence)
    
    def _extract_entities(self, text: str, intent: Intent) -> Dict[str, Any]:
        """Extract entities based on detected intent."""
        entities = {}
        
        # Extract file paths
        file_patterns = [
            r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']',  # Quoted file paths
            r'(\w+\.[a-zA-Z0-9]+)',  # Simple file names
            r'([./\w-]+/[./\w-]+\.[a-zA-Z0-9]+)',  # Path-like structures
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, text)
            if matches:
                entities['files'] = matches
                break
        
        # Extract programming languages
        detected_languages = []
        for lang, keywords in self.language_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                detected_languages.append(lang)
        
        if detected_languages:
            entities['languages'] = detected_languages
        
        # Extract function/class names
        function_pattern = r'(?:function|def|class)\s+(\w+)'
        function_matches = re.findall(function_pattern, text, re.IGNORECASE)
        if function_matches:
            entities['functions'] = function_matches
        
        # Extract quoted strings (potential code snippets or descriptions)
        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted_matches = re.findall(quoted_pattern, text)
        if quoted_matches:
            entities['quoted_text'] = quoted_matches
        
        # Extract numbers (line numbers, etc.)
        number_pattern = r'\b(\d+)\b'
        numbers = re.findall(number_pattern, text)
        if numbers:
            entities['numbers'] = [int(n) for n in numbers]
        
        return entities
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better AI understanding."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize common variations
        replacements = {
            r'\bmujhe\b': 'I need',
            r'\bchahiye\b': 'want',
            r'\bbanana\b': 'to create',
            r'\bkarna\b': 'to do',
            r'\bdikhao\b': 'show',
            r'\bsamjhao\b': 'explain',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def generate_code_from_description(self, description: str, language: str = 'python') -> str:
        """
        Generate code from natural language description.
        
        Args:
            description: Natural language description of what to code
            language: Target programming language
            
        Returns:
            Generated code string
        """
        # This is a simplified version - in a real implementation,
        # this would use more sophisticated NLP models
        
        templates = {
            'python': {
                'function': 'def {name}({params}):\n    """{description}"""\n    pass\n',
                'class': 'class {name}:\n    """{description}"""\n    \n    def __init__(self):\n        pass\n',
                'script': '#!/usr/bin/env python3\n"""{description}"""\n\n# TODO: Implement functionality\npass\n',
            },
            'javascript': {
                'function': 'function {name}({params}) {{\n    // {description}\n    // TODO: Implement\n}}\n',
                'class': 'class {name} {{\n    // {description}\n    constructor() {{\n        // TODO: Implement\n    }}\n}}\n',
                'script': '// {description}\n\n// TODO: Implement functionality\n',
            }
        }
        
        # Extract entities from description
        result = self.process_natural_language(description)
        entities = result.entities
        
        # Determine code type and generate
        if 'function' in description.lower():
            template = templates.get(language, {}).get('function', '')
            name = entities.get('functions', ['myFunction'])[0]
            return template.format(name=name, params='', description=description)
        elif 'class' in description.lower():
            template = templates.get(language, {}).get('class', '')
            name = entities.get('functions', ['MyClass'])[0]
            return template.format(name=name, description=description)
        else:
            template = templates.get(language, {}).get('script', '')
            return template.format(description=description)
    
    def extract_code_intent(self, text: str) -> Dict[str, Any]:
        """Extract specific coding intent and parameters."""
        intent_result = self.process_natural_language(text)
        
        code_intent = {
            'action': intent_result.intent.value,
            'confidence': intent_result.confidence,
            'parameters': intent_result.entities,
            'language': self._detect_primary_language(text),
            'complexity': self._estimate_complexity(text),
        }
        
        return code_intent
    
    def _detect_primary_language(self, text: str) -> Optional[str]:
        """Detect the primary programming language mentioned in text."""
        language_mentions = {}
        
        for lang, keywords in self.language_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text.lower())
            if count > 0:
                language_mentions[lang] = count
        
        if language_mentions:
            return max(language_mentions, key=language_mentions.get)
        
        return None
    
    def _estimate_complexity(self, text: str) -> str:
        """Estimate the complexity of the requested task."""
        complexity_indicators = {
            'simple': ['simple', 'basic', 'easy', 'quick', 'small'],
            'medium': ['medium', 'moderate', 'standard', 'normal'],
            'complex': ['complex', 'advanced', 'difficult', 'large', 'comprehensive', 'full']
        }
        
        text_lower = text.lower()
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return complexity
        
        # Default complexity based on text length and technical terms
        if len(text) > 200 or len(re.findall(r'\b(?:class|function|method|algorithm|database|api|framework)\b', text_lower)) > 3:
            return 'complex'
        elif len(text) > 50:
            return 'medium'
        else:
            return 'simple'
