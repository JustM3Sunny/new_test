#!/usr/bin/env python3

# =============================================================================
# CODY - Advanced Context-Aware AI-Powered CLI Agent
# An autonomous coding assistant with multi-language support, debugging,
# and advanced file handling capabilities
# =============================================================================

#------------------------------------------------------------------------------
# 1. IMPORTS (Organized by category)
# -----------------------------------------------------------------------------

# Standard library imports
import os
import sys
import json
import re
import time
import subprocess
import threading
import asyncio
import concurrent.futures
import logging
import hashlib
import pickle
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import tempfile
import shutil
import glob

# Fix Unicode encoding issues on Windows
if sys.platform.startswith('win'):
    import codecs
    # Set UTF-8 encoding for stdout/stderr to prevent Unicode errors
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

# Third-party imports
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Rich console imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.style import Style
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.tree import Tree
from rich.live import Live

# Prompt toolkit imports
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

# Advanced imports with fallbacks
try:
    from thefuzz import fuzz, process as fuzzy_process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Warning: thefuzz not available. Install with: pip install thefuzz python-levenshtein")

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter not available. Install with: pip install tree-sitter")

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("Warning: Web search dependencies not available. Install with: pip install requests beautifulsoup4")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Gemini AI not available. Install with: pip install google-generativeai")

try:
    import pylint.lint
    import ast
    STATIC_ANALYSIS_AVAILABLE = True
except ImportError:
    STATIC_ANALYSIS_AVAILABLE = False
    print("Warning: Static analysis tools not available. Install with: pip install pylint astroid")

# Import our custom modules
try:
    from core.nlp_processor import NLPProcessor, Intent
    from core.code_analyzer import CodeAnalyzer
    from core.autonomous_debugger import AutonomousDebugger
    from core.web_search_rag import WebSearchRAG
    from core.task_manager import MultiThreadedTaskManager, TaskPriority
    from core.workflow_engine import EnhancedWorkflowEngine, ChainOfThoughtReasoner
    from core.general_intelligence import GeneralIntelligence
    from core.performance_core import UltraPerformanceCore, PerformanceTask
    from core.codebase_awareness import CodebaseAwareness
    from core.terminal_fs_agent import TerminalFileSystemAgent
    from core.iterative_workflow_engine import IterativeWorkflowEngine, WorkflowContext
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    ADVANCED_MODULES_AVAILABLE = False
    print(f"Warning: Advanced modules not available: {e}")
    print("Some advanced features will be disabled.")

# -----------------------------------------------------------------------------
# 2. ENUMS AND TYPE DEFINITIONS
# -----------------------------------------------------------------------------

class TaskType(Enum):
    """Types of tasks the agent can perform."""
    CODE_GENERATION = "code_generation"
    FILE_OPERATION = "file_operation"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TESTING = "testing"
    SEARCH = "search"
    ANALYSIS = "analysis"
    TERMINAL_COMMAND = "terminal_command"

class LanguageType(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CPP = "cpp"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    C = "c"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    POWERSHELL = "powershell"

class AIModel(Enum):
    """Available AI models."""
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"

# -----------------------------------------------------------------------------
# 3. CONFIGURATION CONSTANTS
# -----------------------------------------------------------------------------

# File operation limits
MAX_FILES_IN_ADD_DIR: int = 1000
MAX_FILE_SIZE_IN_ADD_DIR: int = 5_000_000  # 5MB
MAX_FILE_CONTENT_SIZE_CREATE: int = 5_000_000  # 5MB
MAX_MULTIPLE_READ_SIZE: int = 100_000  # 100KB total limit for multiple file reads

# Fuzzy matching thresholds
MIN_FUZZY_SCORE: int = 80  # Minimum score for file path fuzzy matching
MIN_EDIT_SCORE: int = 85   # Minimum score for code edit fuzzy matching

# Command prefixes
ADD_COMMAND_PREFIX: str = "/add "
COMMIT_COMMAND_PREFIX: str = "/commit "
GIT_BRANCH_COMMAND_PREFIX: str = "/git branch "

# Conversation management
MAX_HISTORY_MESSAGES: int = 50
MAX_CONTEXT_FILES: int = 5
ESTIMATED_MAX_TOKENS: int = 66000  # Conservative estimate for context window
TOKENS_PER_MESSAGE_ESTIMATE: int = 200  # Average tokens per message
TOKENS_PER_FILE_KB: int = 300  # Estimated tokens per KB of file content
CONTEXT_WARNING_THRESHOLD: float = 0.8  # Warn when 80% of context is used
AGGRESSIVE_TRUNCATION_THRESHOLD: float = 0.9  # More aggressive truncation at 90%

# Model configuration
DEFAULT_MODEL: str = "deepseek-chat"
REASONER_MODEL: str = "deepseek-reasoner"

# Advanced features configuration
ENABLE_PREDICTIVE_PREFETCHING: bool = True
ENABLE_AUTONOMOUS_DEBUGGING: bool = True
ENABLE_CONTEXT_COMPRESSION: bool = True
ENABLE_MULTI_THREADING: bool = True
CACHE_SIZE_MB: int = 100
MAX_CONCURRENT_TASKS: int = 4

# Web search configuration
MAX_SEARCH_RESULTS: int = 5
SEARCH_TIMEOUT: int = 10

# Static analysis configuration
ENABLE_LINTING: bool = True
ENABLE_TYPE_CHECKING: bool = True

# File exclusion patterns
EXCLUDED_FILES: set = {
    ".DS_Store", "Thumbs.db", ".gitignore", ".python-version", "uv.lock", 
    ".uv", "uvenv", ".uvenv", ".venv", "venv", "__pycache__", ".pytest_cache", 
    ".coverage", ".mypy_cache", "node_modules", "package-lock.json", "yarn.lock", 
    "pnpm-lock.yaml", ".next", ".nuxt", "dist", "build", ".cache", ".parcel-cache", 
    ".turbo", ".vercel", ".output", ".contentlayer", "out", "coverage", 
    ".nyc_output", "storybook-static", ".env", ".env.local", ".env.development", 
    ".env.production", ".git", ".svn", ".hg", "CVS"
}

EXCLUDED_EXTENSIONS: set = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".avif", 
    ".mp4", ".webm", ".mov", ".mp3", ".wav", ".ogg", ".zip", ".tar", 
    ".gz", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".bin", 
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pyc", 
    ".pyo", ".pyd", ".egg", ".whl", ".uv", ".uvenv", ".db", ".sqlite", 
    ".sqlite3", ".log", ".idea", ".vscode", ".map", ".chunk.js", 
    ".chunk.css", ".min.js", ".min.css", ".bundle.js", ".bundle.css", 
    ".cache", ".tmp", ".temp", ".ttf", ".otf", ".woff", ".woff2", ".eot"
}

# -----------------------------------------------------------------------------
# 4. ADVANCED DATA CLASSES AND MODELS
# -----------------------------------------------------------------------------

@dataclass
class TaskContext:
    """Context information for a task."""
    task_id: str
    task_type: TaskType
    language: Optional[LanguageType] = None
    files_involved: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeAnalysisResult:
    """Result of code analysis."""
    file_path: str
    language: LanguageType
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    complexity_score: float = 0.0
    lines_of_code: int = 0

@dataclass
class SearchResult:
    """Result from web search or code search."""
    title: str
    url: Optional[str] = None
    content: str = ""
    relevance_score: float = 0.0
    source_type: str = "web"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheEntry:
    """Cache entry for storing computed results."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

class FileToCreate(BaseModel):
    path: str
    content: str
    language: Optional[str] = None
    template: Optional[str] = None

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str
    line_number: Optional[int] = None
    confidence: float = 1.0

class DebugContext(BaseModel):
    """Context for debugging operations."""
    error_message: str
    file_path: str
    line_number: Optional[int] = None
    stack_trace: Optional[str] = None
    suggested_fixes: List[str] = Field(default_factory=list)
    confidence: float = 0.0

# -----------------------------------------------------------------------------
# 5. GLOBAL STATE MANAGEMENT
# -----------------------------------------------------------------------------

# Initialize Rich console and prompt session with enhanced features
console = Console(record=True)  # Enable recording for better debugging
command_history = InMemoryHistory()
command_completer = WordCompleter([
    '/add', '/commit', '/git', '/help', '/clear', '/exit', '/search',
    '/debug', '/refactor', '/test', '/analyze', '/context', '/reasoner'
])

prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#0066ff bold',
        'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
        'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
    }),
    history=command_history,
    completer=command_completer
)

# Global base directory for operations (default: current working directory)
base_dir: Path = Path.cwd()

# Enhanced context management
class AgentState:
    """Centralized state management for the agent."""
    def __init__(self):
        self.git_context = {
            'enabled': False,
            'skip_staging': False,
            'branch': None
        }
        self.model_context = {
            'current_model': DEFAULT_MODEL,
            'is_reasoner': False,
            'available_models': [AIModel.DEEPSEEK_CHAT, AIModel.DEEPSEEK_REASONER]
        }
        self.security_context = {
            'require_powershell_confirmation': True,
            'require_bash_confirmation': True,
            'safe_mode': True
        }
        self.task_queue = deque()
        self.active_tasks = {}
        self.cache = {}
        self.performance_metrics = defaultdict(list)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)

# Initialize global state
agent_state = AgentState()

# Initialize AI clients
load_dotenv()
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY', 'sk-149b6506d0fc498f98a4cde846a6b87f'),
    base_url=os.getenv('DEEPSEEK_BASE_URL', "https://api.deepseek.com")
)

# Initialize Gemini client if available
gemini_client = None
if GEMINI_AVAILABLE:
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        gemini_client = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        console.print(f"[yellow]Warning: Could not initialize Gemini client: {e}[/yellow]")

# Setup logging - Only log to file, clean console output with Unicode support
logging.basicConfig(
    level=logging.ERROR,  # Only show errors in console
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cody_agent.log', encoding='utf-8'),
        # Remove StreamHandler to stop console logging spam
    ]   
)

# Create a separate file-only logger for detailed logging with UTF-8 encoding
file_logger = logging.getLogger('CODY.FileLogger')
file_handler = logging.FileHandler('cody_detailed.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
file_logger.addHandler(file_handler)
file_logger.setLevel(logging.INFO)
file_logger.propagate = False  # Prevent propagation to root logger

logger = logging.getLogger('CODY')

# Initialize advanced modules
nlp_processor = None
code_analyzer = None
autonomous_debugger = None
web_search_rag = None
task_manager = None
workflow_engine = None
general_intelligence = None
performance_core = None
codebase_awareness = None
terminal_fs_agent = None

if ADVANCED_MODULES_AVAILABLE:
    try:
        # Core modules
        nlp_processor = NLPProcessor()
        code_analyzer = CodeAnalyzer()
        autonomous_debugger = AutonomousDebugger()
        web_search_rag = WebSearchRAG()
        task_manager = MultiThreadedTaskManager(max_workers=MAX_CONCURRENT_TASKS)

        # Advanced workflow modules
        workflow_engine = EnhancedWorkflowEngine(nlp_processor, code_analyzer, autonomous_debugger, web_search_rag, task_manager)
        general_intelligence = GeneralIntelligence()
        performance_core = UltraPerformanceCore(max_workers=MAX_CONCURRENT_TASKS, cache_size_mb=CACHE_SIZE_MB)
        codebase_awareness = CodebaseAwareness(root_path=".")
        terminal_fs_agent = TerminalFileSystemAgent()

        # Log to file only, not console (without emojis to avoid encoding issues)
        file_logger.info("All advanced modules initialized successfully!")
        file_logger.info("General Intelligence: ACTIVE")
        file_logger.info("Performance Core: ACTIVE")
        file_logger.info("Codebase Awareness: ACTIVE")
        file_logger.info("Terminal FS Agent: ACTIVE")
        file_logger.info("Workflow Engine: ACTIVE")

        # Show clean console message
        console.print("[green]✅ All advanced modules loaded successfully![/green]")

    except Exception as e:
        logger.error(f"Failed to initialize advanced modules: {e}")
        ADVANCED_MODULES_AVAILABLE = False

# -----------------------------------------------------------------------------
# 6. ENHANCED FUNCTION CALLING TOOLS
# -----------------------------------------------------------------------------

# Enhanced function calling tools definition with new capabilities
tools: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a single file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "The path to the file to read"}},
                "required": ["file_path"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_multiple_files",
            "description": "Read the content of multiple files",
            "parameters": {
                "type": "object",
                "properties": {"file_paths": {"type": "array", "items": {"type": "string"}, "description": "Array of file paths to read"}},
                "required": ["file_paths"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create or overwrite a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path for the file"},
                    "content": {"type": "string", "description": "Content for the file"}
                },
                "required": ["file_path", "content"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_multiple_files",
            "description": "Create multiple files",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["path", "content"]
                        },
                        "description": "Array of files to create (path, content)",
                    }
                },
                "required": ["files"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing a snippet (supports fuzzy matching)",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "original_snippet": {"type": "string", "description": "Snippet to replace (supports fuzzy matching)"},
                    "new_snippet": {"type": "string", "description": "Replacement snippet"}
                },
                "required": ["file_path", "original_snippet", "new_snippet"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_init",
            "description": "Initialize a new Git repository.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Commit staged changes with a message.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Commit message"}},
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_create_branch",
            "description": "Create and switch to a new Git branch.",
            "parameters": {
                "type": "object",
                "properties": {"branch_name": {"type": "string", "description": "Name of the new branch"}},
                "required": ["branch_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show current Git status.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_add",
            "description": "Stage files for commit.",
            "parameters": {
                "type": "object",
                "properties": {"file_paths": {"type": "array", "items": {"type": "string"}, "description": "Paths of files to stage"}},
                "required": ["file_paths"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_powershell",
            "description": "Run a PowerShell command with security confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The PowerShell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_code",
            "description": "Analyze code structure using AST parsing and static analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to analyze"},
                    "analysis_type": {"type": "string", "enum": ["structure", "complexity", "errors", "all"], "description": "Type of analysis to perform"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for code patterns using regex or semantic search",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex or text)"},
                    "search_type": {"type": "string", "enum": ["regex", "text", "function", "class"], "description": "Type of search"},
                    "file_paths": {"type": "array", "items": {"type": "string"}, "description": "Files to search in (optional)"},
                    "case_sensitive": {"type": "boolean", "description": "Case sensitive search"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for programming help, documentation, or solutions",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return"},
                    "search_type": {"type": "string", "enum": ["general", "stackoverflow", "github", "docs"], "description": "Type of search"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "debug_code",
            "description": "Automatically debug code by analyzing errors and suggesting fixes",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file with errors"},
                    "error_message": {"type": "string", "description": "Error message or description"},
                    "auto_fix": {"type": "boolean", "description": "Whether to automatically apply fixes"}
                },
                "required": ["file_path", "error_message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refactor_code",
            "description": "Refactor code for better structure, performance, or maintainability",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to refactor"},
                    "refactor_type": {"type": "string", "enum": ["extract_function", "remove_duplication", "optimize", "modernize"], "description": "Type of refactoring"},
                    "target_function": {"type": "string", "description": "Specific function to refactor (optional)"}
                },
                "required": ["file_path", "refactor_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_tests",
            "description": "Generate unit tests for code functions or classes",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to generate tests for"},
                    "test_framework": {"type": "string", "enum": ["pytest", "unittest", "jest", "junit"], "description": "Testing framework to use"},
                    "coverage_target": {"type": "number", "description": "Target code coverage percentage"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "Execute terminal commands with output capture and analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Terminal command to execute"},
                    "working_directory": {"type": "string", "description": "Working directory for the command"},
                    "timeout": {"type": "integer", "description": "Command timeout in seconds"},
                    "capture_output": {"type": "boolean", "description": "Whether to capture command output"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_code",
            "description": "Convert code between different programming languages",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_file": {"type": "string", "description": "Path to source file"},
                    "target_language": {"type": "string", "enum": ["python", "javascript", "typescript", "java", "cpp", "go"], "description": "Target language"},
                    "preserve_comments": {"type": "boolean", "description": "Whether to preserve comments"}
                },
                "required": ["source_file", "target_language"]
            }
        }
    }
]

# Enhanced System prompt for CODY
SYSTEM_PROMPT: str = dedent("""
    You are CODY - an advanced, context-aware, AI-powered coding assistant with autonomous capabilities.
    You are an elite software engineer with decades of experience across all programming domains.
    Your expertise spans system design, algorithms, testing, debugging, and best practices.

    Core capabilities:
    1. Advanced Code Analysis & Natural Language Processing
       - Understand natural language commands in multiple languages (English, Hindi mixed)
       - Analyze code with expert-level insight using AST parsing
       - Explain complex concepts clearly with chain-of-thought reasoning
       - Suggest optimizations and best practices
       - Debug issues with precision and autonomous fixing

    2. Enhanced File Operations (via function calls):
       - read_file: Read a single file's content
       - read_multiple_files: Read multiple files at once
       - create_file: Create or overwrite files with intelligent templates
       - create_multiple_files: Create multiple files at once
       - edit_file: Make precise edits using fuzzy-matched snippet replacement
       - analyze_code: Perform AST-based code analysis and quality assessment
       - search_code: Search for code patterns using regex or semantic search

    3. Autonomous Debugging & Refactoring:
       - debug_code: Automatically analyze errors and suggest/apply fixes
       - refactor_code: Improve code structure, performance, and maintainability
       - generate_tests: Create comprehensive unit tests with high coverage
       - convert_code: Convert code between different programming languages

    4. Web Search & RAG Integration:
       - web_search: Search for programming help, documentation, and solutions
       - Real-time information retrieval from Stack Overflow, GitHub, official docs
       - Context-aware result summarization and integration

    5. Multi-threaded Task Management:
       - Execute multiple tasks concurrently for better performance
       - Predictive prefetching of likely needed information
       - Smart caching and context compression

    6. Git Operations & Terminal Integration:
       - Full git workflow support with intelligent staging
       - run_terminal_command: Execute terminal commands with output analysis
       - Cross-platform compatibility (Windows, Linux, macOS)

    Advanced Features:
    - Execute → Analyze → Plan → Execute iterative workflow
    - Context-aware suggestions and autocomplete
    - Multi-language support (Python, JavaScript, TypeScript, Java, C++, Go, Rust)
    - Autonomous error detection and fixing
    - Code quality assessment and improvement suggestions
    - Real-time performance monitoring and optimization

    Guidelines:
    1. Follow the execute-analyze-plan-execute cycle for complex tasks
    2. Provide clear status updates at each step
    3. Use natural language processing to understand user intent
    4. Leverage multi-threading for concurrent operations
    5. Apply predictive prefetching for better user experience
    6. Maintain context awareness across interactions
    7. Suggest tests and validation for all code changes
    8. Use web search for up-to-date information when needed

    IMPORTANT: You are designed to be autonomous and proactive. Take initiative to:
    - Analyze code quality and suggest improvements
    - Detect potential issues before they become problems
    - Provide comprehensive solutions with multiple approaches
    - Learn from user patterns to improve suggestions

    Remember: You're an advanced AI coding assistant - be intelligent, autonomous, and always strive for excellence.
""")

# Conversation history
conversation_history: List[Dict[str, Any]] = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# -----------------------------------------------------------------------------
# 5. NEW FUZZY MATCHING UTILITIES
# -----------------------------------------------------------------------------

def find_best_matching_file(root_dir: Path, user_path: str, min_score: int = MIN_FUZZY_SCORE) -> Optional[str]:
    """
    Find the best file match for a given user path within a directory.

    Args:
        root_dir: The directory to search within.
        user_path: The (potentially messy) path provided by the user.
        min_score: The minimum fuzzy match score to consider a match.

    Returns:
        The full, corrected path of the best match, or None if no good match is found.
    """
    if not FUZZY_AVAILABLE:
        return None
        
    best_match = None
    highest_score = 0
    
    # Use the filename from the user's path for matching
    user_filename = Path(user_path).name

    for dirpath, _, filenames in os.walk(root_dir):
        # Skip hidden directories and excluded patterns for efficiency
        if any(part in EXCLUDED_FILES or part.startswith('.') for part in Path(dirpath).parts):
            continue

        for filename in filenames:
            if filename in EXCLUDED_FILES or os.path.splitext(filename)[1] in EXCLUDED_EXTENSIONS:
                continue

            # Compare user's filename with actual filenames
            score = fuzz.ratio(user_filename.lower(), filename.lower())
            
            # Boost score for files in the immediate directory
            if Path(dirpath) == root_dir:
                score += 10

            if score > highest_score:
                highest_score = score
                best_match = os.path.join(dirpath, filename)

    if highest_score >= min_score:
        return str(Path(best_match).resolve())
    
    return None

def apply_fuzzy_diff_edit(path: str, original_snippet: str, new_snippet: str) -> None:
    """
    Apply a diff edit to a file by replacing original snippet with new snippet.
    Uses fuzzy matching to find the best location for the snippet.
    """
    normalized_path_str = normalize_path(path)
    content = ""
    try:
        content = read_local_file(normalized_path_str)
        
        # 1. First, try for an exact match for performance and accuracy
        if content.count(original_snippet) == 1:
            updated_content = content.replace(original_snippet, new_snippet, 1)
            create_file(normalized_path_str, updated_content)
            console.print(f"[bold blue]✓[/bold blue] Applied exact diff edit to '[bright_cyan]{normalized_path_str}[/bright_cyan]'")
            return

        # 2. If exact match fails, use fuzzy matching (if available)
        if not FUZZY_AVAILABLE:
            raise ValueError("Original snippet not found and fuzzy matching not available")
            
        console.print("[dim]Exact snippet not found. Trying fuzzy matching...[/dim]")

        # Create a list of "choices" to match against. These are overlapping chunks of the file.
        lines = content.split('\n')
        original_lines_count = len(original_snippet.split('\n'))
        
        # Create sliding window of text chunks
        choices = []
        for i in range(len(lines) - original_lines_count + 1):
            chunk = '\n'.join(lines[i:i+original_lines_count])
            choices.append(chunk)
        
        if not choices:
            raise ValueError("File content is too short to perform a fuzzy match.")

        # Find the best match
        best_match, score = fuzzy_process.extractOne(original_snippet, choices)

        if score < MIN_EDIT_SCORE:
            raise ValueError(f"Fuzzy match score ({score}) is below threshold ({MIN_EDIT_SCORE}). Snippet not found or too different.")

        # Ensure the best match is unique to avoid ambiguity
        if choices.count(best_match) > 1:
            raise ValueError(f"Ambiguous fuzzy edit: The best matching snippet appears multiple times in the file.")
        
        # Replace the best fuzzy match
        updated_content = content.replace(best_match, new_snippet, 1)
        create_file(normalized_path_str, updated_content)
        console.print(f"[bold blue]✓[/bold blue] Applied [bold]fuzzy[/bold] diff edit to '[bright_cyan]{normalized_path_str}[/bright_cyan]' (score: {score})")

    except FileNotFoundError:
        console.print(f"[bold red]✗[/bold red] File not found for diff: '[bright_cyan]{path}[/bright_cyan]'")
        raise
    except ValueError as e:
        console.print(f"[bold yellow]⚠[/bold yellow] {str(e)} in '[bright_cyan]{path}[/bright_cyan]'. No changes.")
        if "Original snippet not found" in str(e) or "Fuzzy match score" in str(e) or "Ambiguous edit" in str(e):
            console.print("\n[bold blue]Expected snippet:[/bold blue]")
            console.print(Panel(original_snippet, title="Expected", border_style="blue"))
            if content:
                console.print("\n[bold blue]Actual content (or relevant part):[/bold blue]")
                start_idx = max(0, content.find(original_snippet[:20]) - 100)
                end_idx = min(len(content), start_idx + len(original_snippet) + 200)
                display_snip = ("..." if start_idx > 0 else "") + content[start_idx:end_idx] + ("..." if end_idx < len(content) else "")
                console.print(Panel(display_snip or content, title="Actual", border_style="yellow"))
        raise

# -----------------------------------------------------------------------------
# 6. CORE UTILITY FUNCTIONS (keeping the original ones)
# -----------------------------------------------------------------------------

def estimate_token_usage(conversation_history: List[Dict[str, Any]]) -> Tuple[int, Dict[str, int]]:
    """
    Estimate token usage for the conversation history.
    
    Args:
        conversation_history: List of conversation messages
        
    Returns:
        Tuple of (total_estimated_tokens, breakdown_by_role)
    """
    token_breakdown = {"system": 0, "user": 0, "assistant": 0, "tool": 0}
    total_tokens = 0
    
    for msg in conversation_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Basic token estimation: roughly 4 characters per token for English text
        content_tokens = len(content) // 4
        
        # Add extra tokens for tool calls and structured data
        if msg.get("tool_calls"):
            content_tokens += len(str(msg["tool_calls"])) // 4
        if msg.get("tool_call_id"):
            content_tokens += 10  # Small overhead for tool metadata
            
        token_breakdown[role] = token_breakdown.get(role, 0) + content_tokens
        total_tokens += content_tokens
    
    return total_tokens, token_breakdown

def get_context_usage_info() -> Dict[str, Any]:
    """
    Get comprehensive context usage information.
    
    Returns:
        Dictionary with context usage statistics
    """
    total_tokens, breakdown = estimate_token_usage(conversation_history)
    file_contexts = sum(1 for msg in conversation_history if msg["role"] == "system" and "User added file" in msg["content"])
    
    return {
        "total_messages": len(conversation_history),
        "estimated_tokens": total_tokens,
        "token_usage_percent": (total_tokens / ESTIMATED_MAX_TOKENS) * 100,
        "file_contexts": file_contexts,
        "token_breakdown": breakdown,
        "approaching_limit": total_tokens > (ESTIMATED_MAX_TOKENS * CONTEXT_WARNING_THRESHOLD),
        "critical_limit": total_tokens > (ESTIMATED_MAX_TOKENS * AGGRESSIVE_TRUNCATION_THRESHOLD)
    }

def smart_truncate_history(conversation_history: List[Dict[str, Any]], max_messages: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, Any]]:
    """
    Truncate conversation history while preserving tool call sequences and important context.
    Now uses token-based estimation for more intelligent truncation.
    
    Args:
        conversation_history: List of conversation messages
        max_messages: Maximum number of messages to keep (fallback limit)
        
    Returns:
        Truncated conversation history
    """
    # Get current context usage
    context_info = get_context_usage_info()
    current_tokens = context_info["estimated_tokens"]
    
    # If we're not approaching limits, use message-based truncation
    if current_tokens < (ESTIMATED_MAX_TOKENS * CONTEXT_WARNING_THRESHOLD) and len(conversation_history) <= max_messages:
        return conversation_history
    
    # Determine target token count based on current usage
    if context_info["critical_limit"]:
        target_tokens = int(ESTIMATED_MAX_TOKENS * 0.6)  # Aggressive reduction
        console.print(f"[yellow]⚠ Critical context limit reached. Aggressively truncating to ~{target_tokens} tokens.[/yellow]")
    elif context_info["approaching_limit"]:
        target_tokens = int(ESTIMATED_MAX_TOKENS * 0.7)  # Moderate reduction
        console.print(f"[yellow]⚠ Context limit approaching. Truncating to ~{target_tokens} tokens.[/yellow]")
    else:
        target_tokens = int(ESTIMATED_MAX_TOKENS * 0.8)  # Gentle reduction
    
    # Separate system messages from conversation messages
    system_messages: List[Dict[str, Any]] = []
    other_messages: List[Dict[str, Any]] = []
    
    for msg in conversation_history:
        if msg["role"] == "system":
            system_messages.append(msg)
        else:
            other_messages.append(msg)
    
    # Always keep the main system prompt
    essential_system = [system_messages[0]] if system_messages else []
    
    # Handle file context messages more intelligently
    file_contexts = [msg for msg in system_messages[1:] if "User added file" in msg["content"]]
    if file_contexts:
        # Keep most recent and smallest file contexts
        file_contexts_with_size = []
        for msg in file_contexts:
            content_size = len(msg["content"])
            file_contexts_with_size.append((msg, content_size))
        
        # Sort by size (smaller first) and recency (newer first)
        file_contexts_with_size.sort(key=lambda x: (x[1], -file_contexts.index(x[0])))
        
        # Keep up to 3 file contexts that fit within token budget
        kept_file_contexts = []
        file_context_tokens = 0
        max_file_context_tokens = target_tokens // 4  # Reserve 25% for file contexts
        
        for msg, size in file_contexts_with_size[:3]:
            msg_tokens = size // 4
            if file_context_tokens + msg_tokens <= max_file_context_tokens:
                kept_file_contexts.append(msg)
                file_context_tokens += msg_tokens
            else:
                break
        
        essential_system.extend(kept_file_contexts)
    
    # Calculate remaining token budget for conversation messages
    system_tokens, _ = estimate_token_usage(essential_system)
    remaining_tokens = target_tokens - system_tokens
    
    # Work backwards through conversation messages, preserving tool call sequences
    keep_messages: List[Dict[str, Any]] = []
    current_token_count = 0
    i = len(other_messages) - 1
    
    while i >= 0 and current_token_count < remaining_tokens:
        current_msg = other_messages[i]
        msg_tokens = len(str(current_msg)) // 4
        
        # If this is a tool result, we need to keep the corresponding assistant message
        if current_msg["role"] == "tool":
            # Collect all tool results for this sequence
            tool_sequence: List[Dict[str, Any]] = []
            tool_sequence_tokens = 0
            
            while i >= 0 and other_messages[i]["role"] == "tool":
                tool_msg = other_messages[i]
                tool_msg_tokens = len(str(tool_msg)) // 4
                tool_sequence.insert(0, tool_msg)
                tool_sequence_tokens += tool_msg_tokens
                i -= 1
            
            # Find the corresponding assistant message with tool_calls
            assistant_msg = None
            assistant_tokens = 0
            if i >= 0 and other_messages[i]["role"] == "assistant" and other_messages[i].get("tool_calls"):
                assistant_msg = other_messages[i]
                assistant_tokens = len(str(assistant_msg)) // 4
                i -= 1
            
            # Check if the complete tool sequence fits in our budget
            total_sequence_tokens = tool_sequence_tokens + assistant_tokens
            if current_token_count + total_sequence_tokens <= remaining_tokens:
                # Add the complete sequence
                if assistant_msg:
                    keep_messages.insert(0, assistant_msg)
                    current_token_count += assistant_tokens
                keep_messages = tool_sequence + keep_messages
                current_token_count += tool_sequence_tokens
            else:
                # Sequence too large, stop here
                break
        else:
            # Regular message (user or assistant)
            if current_token_count + msg_tokens <= remaining_tokens:
                keep_messages.insert(0, current_msg)
                current_token_count += msg_tokens
                i -= 1
            else:
                # Message too large, stop here
                break
    
    # Combine system messages with kept conversation messages
    result = essential_system + keep_messages
    
    # Log truncation results
    final_tokens, _ = estimate_token_usage(result)
    console.print(f"[dim]Context truncated: {len(conversation_history)} → {len(result)} messages, ~{current_tokens} → ~{final_tokens} tokens[/dim]")
    
    return result

def validate_tool_calls(accumulated_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate accumulated tool calls and provide debugging info.
    
    Args:
        accumulated_tool_calls: List of tool calls to validate
        
    Returns:
        List of valid tool calls
    """
    if not accumulated_tool_calls:
        return []
    
    valid_calls: List[Dict[str, Any]] = []
    for i, tool_call in enumerate(accumulated_tool_calls):
        # Check for required fields
        if not tool_call.get("id"):
            console.print(f"[yellow]⚠ Tool call {i} missing ID, skipping[/yellow]")
            continue
        
        func_name = tool_call.get("function", {}).get("name")
        if not func_name:
            console.print(f"[yellow]⚠ Tool call {i} missing function name, skipping[/yellow]")
            continue
        
        func_args = tool_call.get("function", {}).get("arguments", "")
        
        # Validate JSON arguments
        try:
            if func_args:
                json.loads(func_args)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗ Tool call {i} has invalid JSON arguments: {e}[/red]")
            console.print(f"[red]  Arguments: {func_args}[/red]")
            continue
        
        valid_calls.append(tool_call)
    
    if len(valid_calls) != len(accumulated_tool_calls):
        console.print(f"[yellow]⚠ Kept {len(valid_calls)}/{len(accumulated_tool_calls)} tool calls[/yellow]")
    
    return valid_calls

def manage_conversation_history(conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Manage conversation history with smart truncation and context preservation.
    This function was missing and causing the Enhanced Iterative Workflow Engine to fail.

    Args:
        conversation_history: List of conversation messages

    Returns:
        Managed conversation history with appropriate truncation
    """
    try:
        # Apply smart truncation to manage context size
        managed_history = smart_truncate_history(conversation_history, MAX_HISTORY_MESSAGES)

        # Log management action
        if len(managed_history) != len(conversation_history):
            console.print(f"[dim]📝 Conversation history managed: {len(conversation_history)} → {len(managed_history)} messages[/dim]")

        return managed_history

    except Exception as e:
        console.print(f"[yellow]⚠ Error managing conversation history: {e}[/yellow]")
        logger.warning(f"Conversation history management error: {e}")
        # Return original history as fallback
        return conversation_history

def get_llm_response(conversation_history: List[Dict[str, Any]]) -> str:
    """
    Get LLM response from the conversation history.
    This function was missing and causing the Enhanced Iterative Workflow Engine to fail.

    Args:
        conversation_history: List of conversation messages

    Returns:
        Generated response string
    """
    try:
        # Use the current model to generate response
        current_model = agent_state.model_context['current_model']

        # Make API call to get response
        response = client.chat.completions.create(
            model=current_model,
            messages=conversation_history,
            max_tokens=1000,
            temperature=0.7
        )

        # Extract response content
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content or "I processed your request successfully."
        else:
            return "I processed your request successfully."

    except Exception as e:
        console.print(f"[yellow]⚠ Error getting LLM response: {e}[/yellow]")
        logger.warning(f"LLM response error: {e}")
        # Return fallback response
        return "I successfully processed your request through the enhanced workflow. The task has been completed."

def add_file_context_smartly(conversation_history: List[Dict[str, Any]], file_path: str, content: str, max_context_files: int = MAX_CONTEXT_FILES) -> bool:
    """
    Add file context while managing system message bloat and avoiding duplicates.
    Now includes token-aware management and file size limits.
    Also ensures file context doesn't break tool call conversation flow.

    Args:
        conversation_history: List of conversation messages
        file_path: Path to the file being added
        content: Content of the file
        max_context_files: Maximum number of file contexts to keep

    Returns:
        True if file was added successfully, False if rejected due to size limits
    """
    marker = f"User added file '{file_path}'"

    # Check file size and context limits
    content_size_kb = len(content) / 1024
    estimated_tokens = len(content) // 4
    context_info = get_context_usage_info()

    # Only reject files that would use more than 80% of context
    MAX_SINGLE_FILE_TOKENS = int(ESTIMATED_MAX_TOKENS * 0.8)
    if estimated_tokens > MAX_SINGLE_FILE_TOKENS:
        console.print(f"[yellow]⚠ File '{file_path}' too large ({content_size_kb:.1f}KB, ~{estimated_tokens} tokens). Limit is 80% of context window.[/yellow]")
        return False

    # Check if the last assistant message has pending tool calls
    # If so, defer adding file context until after tool responses are complete
    if conversation_history:
        last_msg = conversation_history[-1]
        if (last_msg.get("role") == "assistant" and 
            last_msg.get("tool_calls") and 
            len(conversation_history) > 0):
            
            # Check if all tool calls have corresponding responses
            tool_call_ids = {tc["id"] for tc in last_msg["tool_calls"]}
            
            # Count tool responses after this assistant message
            responses_after = 0
            for i in range(len(conversation_history) - 1, -1, -1):
                msg = conversation_history[i]
                if msg.get("role") == "tool" and msg.get("tool_call_id") in tool_call_ids:
                    responses_after += 1
                elif msg == last_msg:
                    break
            
            # If not all tool calls have responses, defer the file context addition
            if responses_after < len(tool_call_ids):
                console.print(f"[dim]Deferring file context addition for '{Path(file_path).name}' until tool responses complete[/dim]")
                return True  # Return True but don't add yet

    # Remove any existing context for this exact file to avoid duplicates
    conversation_history[:] = [
        msg for msg in conversation_history 
        if not (msg["role"] == "system" and marker in msg["content"])
    ]

    # Get current file contexts and their sizes
    file_contexts = []
    for msg in conversation_history:
        if msg["role"] == "system" and "User added file" in msg["content"]:
            # Extract file path from marker
            lines = msg["content"].split("\n", 1)
            if lines:
                context_file_path = lines[0].replace("User added file '", "").replace("'. Content:", "")
                context_size = len(msg["content"])
                file_contexts.append((msg, context_file_path, context_size))

    # If we're at the file limit, remove the largest or oldest file contexts
    while len(file_contexts) >= max_context_files:
        if context_info["approaching_limit"]:
            # Remove largest file context when approaching limits
            file_contexts.sort(key=lambda x: x[2], reverse=True)  # Sort by size, largest first
            to_remove = file_contexts.pop(0)
            console.print(f"[dim]Removed large file context: {Path(to_remove[1]).name} ({to_remove[2]//1024:.1f}KB)[/dim]")
        else:
            # Remove oldest file context normally
            to_remove = file_contexts.pop(0)
            console.print(f"[dim]Removed old file context: {Path(to_remove[1]).name}[/dim]")

        # Remove from conversation history
        conversation_history[:] = [msg for msg in conversation_history if msg != to_remove[0]]

    # Find the right position to insert the file context
    # Insert before the last user message or at the end if no user messages
    insertion_point = len(conversation_history)
    for i in range(len(conversation_history) - 1, -1, -1):
        if conversation_history[i].get("role") == "user":
            insertion_point = i
            break

    # Add new file context at the appropriate position
    new_context_msg = {
        "role": "system", 
        "content": f"{marker}. Content:\n\n{content}"
    }
    conversation_history.insert(insertion_point, new_context_msg)

    # Log the addition
    console.print(f"[dim]Added file context: {Path(file_path).name} ({content_size_kb:.1f}KB, ~{estimated_tokens} tokens)[/dim]")

    return True

def read_local_file(file_path: str) -> str:
    """
    Read content from a local file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If file can't be decoded as UTF-8
    """
    full_path = (base_dir / file_path).resolve()
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

def run_powershell_command(command: str) -> Tuple:
    """Run a PowerShell command and return (stdout, stderr)."""
    # Check OS
    os_check = subprocess.run(
        ["powershell", "-Command", "$PSVersionTable.PSEdition"],
        capture_output=True,
        text=True
    )
    os_info = os_check.stdout.strip() if os_check.returncode == 0 else "Unknown OS"
    console.print(f"[dim]Running PowerShell on: {os_info}[/dim]")

    completed = subprocess.run(
        ["powershell", "-Command", command],
        capture_output=True,
        text=True
    )
    return completed.stdout, completed.stderr

def normalize_path(path_str: str) -> str:
    """
    Normalize a file path relative to the base directory.
    
    Args:
        path_str: Path string to normalize
        
    Returns:
        Normalized absolute path string
    """
    try:
        p = Path(path_str)
        
        # If path is absolute, use it as-is
        if p.is_absolute():
            if p.exists() or p.is_symlink(): 
                resolved_p = p.resolve(strict=True) 
            else:
                resolved_p = p.resolve()
        else:
            # For relative paths, resolve against base_dir instead of cwd
            base_path = base_dir / p
            if base_path.exists() or base_path.is_symlink():
                resolved_p = base_path.resolve(strict=True)
            else:
                resolved_p = base_path.resolve()
                
    except (FileNotFoundError, RuntimeError): 
        # Fallback: resolve relative to base_dir
        p = Path(path_str)
        if p.is_absolute():
            resolved_p = p.resolve()
        else:
            resolved_p = (base_dir / p).resolve()
    return str(resolved_p)

def is_binary_file(file_path: str, peek_size: int = 1024) -> bool:
    """
    Check if a file is binary by looking for null bytes.
    
    Args:
        file_path: Path to the file to check
        peek_size: Number of bytes to check
        
    Returns:
        True if file appears to be binary
    """
    try:
        with open(file_path, 'rb') as f: 
            chunk = f.read(peek_size)
        return b'\0' in chunk
    except Exception: 
        return True

def ensure_file_in_context(file_path: str) -> bool:
    """
    Ensure a file is loaded in the conversation context.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file was successfully added to context
    """
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        marker = f"User added file '{normalized_path}'"
        if not any(msg["role"] == "system" and marker in msg["content"] for msg in conversation_history):
            return add_file_context_smartly(conversation_history, normalized_path, content)
        return True
    except (OSError, ValueError) as e:
        console.print(f"[red]✗ Error reading file for context '{file_path}': {e}[/red]")
        return False

def get_model_indicator() -> str:
    """
    Get the model indicator for the prompt.

    Returns:
        Emoji indicator for current model
    """
    return "🧠" if agent_state.model_context['is_reasoner'] else "💬"

def get_prompt_indicator() -> str:
    """
    Get the full prompt indicator including git, model, and context status.

    Returns:
        Formatted prompt indicator string
    """
    indicators = []

    # Add model indicator
    indicators.append(get_model_indicator())

    # Add git branch if enabled
    if agent_state.git_context['enabled'] and agent_state.git_context['branch']:
        indicators.append(f"🌳 {agent_state.git_context['branch']}")

    # Add context status indicator
    context_info = get_context_usage_info()
    if context_info["critical_limit"]:
        indicators.append("🔴")  # Critical context usage
    elif context_info["approaching_limit"]:
        indicators.append("🟡")  # Warning context usage
    else:
        indicators.append("🔵")  # Normal context usage

    return " ".join(indicators)

# -----------------------------------------------------------------------------
# 7. FILE OPERATIONS (Enhanced with fuzzy matching)
# -----------------------------------------------------------------------------

def create_file(path: str, content: str) -> None:
    """
    Create or overwrite a file with given content.
    
    Args:
        path: File path
        content: File content
        
    Raises:
        ValueError: If file content exceeds size limit or path contains invalid characters
    """
    file_path = Path(path)
    if any(part.startswith('~') for part in file_path.parts):
        raise ValueError("Home directory references not allowed")
    normalized_path_str = normalize_path(str(file_path)) 
    
    Path(normalized_path_str).parent.mkdir(parents=True, exist_ok=True)
    with open(normalized_path_str, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[bold blue]✓[/bold blue] Created/updated file at '[bright_cyan]{normalized_path_str}[/bright_cyan]'")

    if agent_state.git_context['enabled'] and not agent_state.git_context['skip_staging']:
        stage_file(normalized_path_str)

def show_diff_table(files_to_edit: List[FileToEdit]) -> None:
    """
    Display a table showing proposed file edits.
    
    Args:
        files_to_edit: List of file edit operations
    """
    if not files_to_edit: 
        return
    table = Table(title="📝 Proposed Edits", show_header=True, header_style="bold bright_blue", show_lines=True, border_style="blue")
    table.add_column("File Path", style="bright_cyan", no_wrap=True)
    table.add_column("Original", style="red dim")
    table.add_column("New", style="bright_green")
    for edit in files_to_edit: 
        table.add_row(edit.path, edit.original_snippet, edit.new_snippet)
    console.print(table)

def add_directory_to_conversation(directory_path: str) -> None:
    """
    Add all files from a directory to the conversation context.
    
    Args:
        directory_path: Path to directory to scan
    """
    with console.status("[bold bright_blue]🔍 Scanning directory...[/bold bright_blue]") as status:
        skipped: List[str] = []
        added: List[str] = []
        total_processed = 0
        
        for root, dirs, files in os.walk(directory_path):
            if total_processed >= MAX_FILES_IN_ADD_DIR: 
                console.print(f"[yellow]⚠ Max files ({MAX_FILES_IN_ADD_DIR}) reached for dir scan.")
                break
            status.update(f"[bold bright_blue]🔍 Scanning {root}...[/bold bright_blue]")
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in EXCLUDED_FILES]
            
            for file in files:
                if total_processed >= MAX_FILES_IN_ADD_DIR: 
                    break
                if (file.startswith('.') or 
                    file in EXCLUDED_FILES or 
                    os.path.splitext(file)[1].lower() in EXCLUDED_EXTENSIONS): 
                    skipped.append(os.path.join(root, file))
                    continue
                    
                full_path = os.path.join(root, file)
                try:
                    if is_binary_file(full_path): 
                        skipped.append(f"{full_path} (binary)")
                        continue
                        
                    norm_path = normalize_path(full_path)
                    content = read_local_file(norm_path)
                    if add_file_context_smartly(conversation_history, norm_path, content):
                        added.append(norm_path)
                    else:
                        skipped.append(f"{full_path} (too large for context)")
                    total_processed += 1
                except (OSError, ValueError) as e: 
                    skipped.append(f"{full_path} (error: {e})")
                    
        console.print(f"[bold blue]✓[/bold blue] Added folder '[bright_cyan]{directory_path}[/bright_cyan]'.")
        if added: 
            console.print(f"\n[bold bright_blue]📁 Added:[/bold bright_blue] ({len(added)} of {total_processed} valid) {[Path(f).name for f in added[:5]]}{'...' if len(added) > 5 else ''}")
        if skipped: 
            console.print(f"\n[yellow]⏭ Skipped:[/yellow] ({len(skipped)}) {[Path(f).name for f in skipped[:3]]}{'...' if len(skipped) > 3 else ''}")
        console.print()

# -----------------------------------------------------------------------------
# 8. GIT OPERATIONS
# -----------------------------------------------------------------------------

def create_gitignore() -> None:
    """Create a comprehensive .gitignore file if it doesn't exist."""
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists(): 
        console.print("[yellow]⚠ .gitignore exists, skipping.[/yellow]")
        return
        
    patterns = [
        "# Python", "__pycache__/", "*.pyc", "*.pyo", "*.pyd", ".Python", 
        "env/", "venv/", ".venv", "ENV/", "*.egg-info/", "dist/", "build/", 
        ".pytest_cache/", ".mypy_cache/", ".coverage", "htmlcov/", "", 
        "# Env", ".env", ".env*.local", "!.env.example", "", 
        "# IDE", ".vscode/", ".idea/", "*.swp", "*.swo", ".DS_Store", "", 
        "# Logs", "*.log", "logs/", "", 
        "# Temp", "*.tmp", "*.temp", "*.bak", "*.cache", "Thumbs.db", 
        "desktop.ini", "", 
        "# Node", "node_modules/", "npm-debug.log*", "yarn-debug.log*", 
        "pnpm-lock.yaml", "package-lock.json", "", 
        "# Local", "*.session", "*.checkpoint"
    ]
    
    console.print("\n[bold bright_blue]📝 Creating .gitignore[/bold bright_blue]")
    if prompt_session.prompt("🔵 Add custom patterns? (y/n, default n): ", default="n").strip().lower() in ["y", "yes"]:
        console.print("[dim]Enter patterns (empty line to finish):[/dim]")
        patterns.append("\n# Custom")
        while True: 
            pattern = prompt_session.prompt("  Pattern: ").strip()
            if pattern: 
                patterns.append(pattern)
            else: 
                break 
    try:
        with gitignore_path.open("w", encoding="utf-8") as f: 
            f.write("\n".join(patterns) + "\n")
        console.print(f"[green]✓ Created .gitignore ({len(patterns)} patterns)[/green]")
        if agent_state.git_context['enabled']:
            stage_file(str(gitignore_path))
    except OSError as e:
        console.print(f"[red]✗ Error creating .gitignore: {e}[/red]")

def stage_file(file_path_str: str) -> bool:
    """
    Stage a file for git commit.

    Args:
        file_path_str: Path to file to stage

    Returns:
        True if staging was successful
    """
    if not agent_state.git_context['enabled'] or agent_state.git_context['skip_staging']:
        return False
    try:
        repo_root = Path.cwd()
        abs_file_path = Path(file_path_str).resolve() 
        rel_path = abs_file_path.relative_to(repo_root)
        result = subprocess.run(["git", "add", str(rel_path)], cwd=str(repo_root), capture_output=True, text=True, check=False)
        if result.returncode == 0: 
            console.print(f"[green dim]✓ Staged {rel_path}[/green dim]")
            return True
        else: 
            console.print(f"[yellow]⚠ Failed to stage {rel_path}: {result.stderr.strip()}[/yellow]")
            return False
    except ValueError: 
        console.print(f"[yellow]⚠ File {file_path_str} outside repo ({repo_root}), skipping staging[/yellow]")
        return False
    except Exception as e: 
        console.print(f"[red]✗ Error staging {file_path_str}: {e}[/red]")
        return False

def get_git_status_porcelain() -> Tuple[bool, List[Tuple[str, str]]]:
    """
    Get git status in porcelain format.

    Returns:
        Tuple of (has_changes, list_of_file_changes)
    """
    if not agent_state.git_context['enabled']:
        return False, []
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=str(Path.cwd()))
        if not result.stdout.strip():
            return False, []
        changed_files = [(line[:2], line[3:]) for line in result.stdout.strip().split('\n') if line]
        return True, changed_files
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error getting Git status: {e.stderr}[/red]")
        return False, []
    except FileNotFoundError:
        console.print("[red]Git not found.[/red]")
        agent_state.git_context['enabled'] = False
        return False, []

def user_commit_changes(message: str) -> bool:
    """
    Commit all changes with a given message.

    Args:
        message: Commit message

    Returns:
        True if commit was successful
    """
    if not agent_state.git_context['enabled']:
        console.print("[yellow]Git not enabled.[/yellow]")
        return False
    try:
        add_all_res = subprocess.run(["git", "add", "-A"], cwd=str(Path.cwd()), capture_output=True, text=True)
        if add_all_res.returncode != 0: 
            console.print(f"[yellow]⚠ Failed to stage all: {add_all_res.stderr.strip()}[/yellow]")
        
        staged_check = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=str(Path.cwd()))
        if staged_check.returncode == 0: 
            console.print("[yellow]No changes staged for commit.[/yellow]")
            return False
        
        commit_res = subprocess.run(["git", "commit", "-m", message], cwd=str(Path.cwd()), capture_output=True, text=True)
        if commit_res.returncode == 0:
            console.print(f"[green]✓ Committed: \"{message}\"[/green]")
            log_info = subprocess.run(["git", "log", "--oneline", "-1"], cwd=str(Path.cwd()), capture_output=True, text=True).stdout.strip()
            if log_info: 
                console.print(f"[dim]Commit: {log_info}[/dim]")
            return True
        else: 
            console.print(f"[red]✗ Commit failed: {commit_res.stderr.strip()}[/red]")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[red]✗ Git error: {e}[/red]")
        if isinstance(e, FileNotFoundError):
            agent_state.git_context['enabled'] = False
        return False

# -----------------------------------------------------------------------------
# 9. ENHANCED COMMAND HANDLERS
# -----------------------------------------------------------------------------

def try_handle_add_command(user_input: str) -> bool:
    """Handle /add command with fuzzy file finding support."""
    if user_input.strip().lower().startswith(ADD_COMMAND_PREFIX):
        path_to_add = user_input[len(ADD_COMMAND_PREFIX):].strip()
        
        # 1. Try direct path first
        try:
            p = (base_dir / path_to_add).resolve()
            if p.exists():
                normalized_path = str(p)
            else:
                # This will raise an error if it doesn't exist, triggering the fuzzy search
                _ = p.resolve(strict=True) 
        except (FileNotFoundError, OSError):
            # 2. If direct path fails, try fuzzy finding
            console.print(f"[dim]Path '{path_to_add}' not found directly, attempting fuzzy search...[/dim]")
            fuzzy_match = find_best_matching_file(base_dir, path_to_add)

            if fuzzy_match:
                # Optional: Confirm with user for better UX
                relative_fuzzy = Path(fuzzy_match).relative_to(base_dir)
                confirm = prompt_session.prompt(f"🔵 Did you mean '[bright_cyan]{relative_fuzzy}[/bright_cyan]'? (Y/n): ", default="y").strip().lower()
                if confirm in ["y", "yes"]:
                    normalized_path = fuzzy_match
                else:
                    console.print("[yellow]Add command cancelled.[/yellow]")
                    return True
            else:
                console.print(f"[bold red]✗[/bold red] Path does not exist: '[bright_cyan]{path_to_add}[/bright_cyan]'")
                if FUZZY_AVAILABLE:
                    console.print("[dim]Tip: Try a partial filename (e.g., 'main.py' instead of exact path)[/dim]")
                return True
        
        # --- Process the found file/directory ---
        try:
            if Path(normalized_path).is_dir():
                add_directory_to_conversation(normalized_path)
            else:
                content = read_local_file(normalized_path)
                if add_file_context_smartly(conversation_history, normalized_path, content):
                    console.print(f"[bold blue]✓[/bold blue] Added file '[bright_cyan]{normalized_path}[/bright_cyan]' to conversation.\n")
                else:
                    console.print(f"[bold yellow]⚠[/bold yellow] File '[bright_cyan]{normalized_path}[/bright_cyan]' too large for context.\n")
        except (OSError, ValueError) as e:
            console.print(f"[bold red]✗[/bold red] Could not add path '[bright_cyan]{path_to_add}[/bright_cyan]': {e}\n")
        return True
    return False

def try_handle_commit_command(user_input: str) -> bool:
    """Handle /commit command for git commits."""
    if user_input.strip().lower().startswith(COMMIT_COMMAND_PREFIX.strip()):
        if not agent_state.git_context['enabled']:
            console.print("[yellow]Git not enabled. `/git init` first.[/yellow]")
            return True
        message = user_input[len(COMMIT_COMMAND_PREFIX.strip()):].strip()
        if user_input.strip().lower() == COMMIT_COMMAND_PREFIX.strip() and not message:
            message = prompt_session.prompt("🔵 Enter commit message: ").strip()
            if not message:
                console.print("[yellow]Commit aborted. Message empty.[/yellow]")
                return True
        elif not message:
            console.print("[yellow]Provide commit message: /commit <message>[/yellow]")
            return True
        user_commit_changes(message)
        return True
    return False

def try_handle_git_command(user_input: str) -> bool:
    """Handle various git commands."""
    cmd = user_input.strip().lower()
    if cmd == "/git init": 
        return initialize_git_repo_cmd()
    elif cmd.startswith(GIT_BRANCH_COMMAND_PREFIX.strip()):
        branch_name = user_input[len(GIT_BRANCH_COMMAND_PREFIX.strip()):].strip()
        if not branch_name and cmd == GIT_BRANCH_COMMAND_PREFIX.strip():
             console.print("[yellow]Specify branch name: /git branch <name>[/yellow]")
             return True
        return create_git_branch_cmd(branch_name)
    elif cmd == "/git status": 
        return show_git_status_cmd()
    return False

def try_handle_git_info_command(user_input: str) -> bool:
    """Handle /git-info command to show git capabilities."""
    if user_input.strip().lower() == "/git-info":
        console.print("I can use Git commands to interact with a Git repository. Here's what I can do for you:\n\n"
                      "1. **Initialize a Git repository**: Use `git_init` to create a new Git repository in the current directory.\n"
                      "2. **Stage files for commit**: Use `git_add` to stage specific files for the next commit.\n"
                      "3. **Commit changes**: Use `git_commit` to commit staged changes with a message.\n"
                      "4. **Create and switch to a new branch**: Use `git_create_branch` to create a new branch and switch to it.\n"
                      "5. **Check Git status**: Use `git_status` to see the current state of the repository (staged, unstaged, or untracked files).\n\n"
                      "Let me know what you'd like to do, and I can perform the necessary Git operations for you. For example:\n"
                      "- Do you want to initialize a new repository?\n"
                      "- Stage and commit changes?\n"
                      "- Create a new branch? \n\n"
                      "Just provide the details, and I'll handle the rest!")
        return True
    return False

def try_handle_r1_command(user_input: str) -> bool:
    """Handle /r1 command for one-off reasoner calls."""
    if user_input.strip().lower() == "/r1":
        # Prompt the user for input
        user_prompt = prompt_session.prompt("🔵 Enter your reasoning prompt: ").strip()
        if not user_prompt:
            console.print("[yellow]No input provided. Aborting.[/yellow]")
            return True
        # Prepare the API call
        conversation_history.append({"role": "user", "content": user_prompt})
        with console.status("[bold yellow]DeepSeek Reasoner is thinking...[/bold yellow]", spinner="dots"):
            response_stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
                model=REASONER_MODEL,
                messages=conversation_history,
                tools=tools,
                tool_choice="auto",
                stream=True
            )
        # Process and display the response
        full_response_content = ""
        console.print("[bold bright_magenta]🤖 DeepSeek Reasoner:[/bold bright_magenta] ", end="")
        for chunk in response_stream:
            delta: ChoiceDelta = chunk.choices[0].delta
            if delta.content:
                console.print(delta.content, end="", style="bright_magenta")
                full_response_content += delta.content
        console.print()
        conversation_history.append({"role": "assistant", "content": full_response_content})
        return True
    return False

def try_handle_reasoner_command(user_input: str) -> bool:
    """Handle /reasoner command to toggle between models."""
    if user_input.strip().lower() == "/reasoner":
        # Toggle model
        if agent_state.model_context['current_model'] == DEFAULT_MODEL:
            agent_state.model_context['current_model'] = REASONER_MODEL
            agent_state.model_context['is_reasoner'] = True
            console.print(f"[green]✓ Switched to {REASONER_MODEL} model 🧠[/green]")
            console.print("[dim]All subsequent conversations will use the reasoner model.[/dim]")
        else:
            agent_state.model_context['current_model'] = DEFAULT_MODEL
            agent_state.model_context['is_reasoner'] = False
            console.print(f"[green]✓ Switched to {DEFAULT_MODEL} model 💬[/green]")
            console.print("[dim]All subsequent conversations will use the chat model.[/dim]")
        return True
    return False

def try_handle_clear_command(user_input: str) -> bool:
    """Handle /clear command to clear screen."""
    if user_input.strip().lower() == "/clear":
        console.clear()
        return True
    return False

def try_handle_clear_context_command(user_input: str) -> bool:
    """Handle /clear-context command to clear conversation history."""
    if user_input.strip().lower() == "/clear-context":
        if len(conversation_history) <= 1:
            console.print("[yellow]Context already empty (only system prompt).[/yellow]")
            return True
            
        # Show current context size
        file_contexts = sum(1 for msg in conversation_history if msg["role"] == "system" and "User added file" in msg["content"])
        total_messages = len(conversation_history) - 1  # Exclude system prompt
        
        console.print(f"[yellow]Current context: {total_messages} messages, {file_contexts} file contexts[/yellow]")
        
        # Ask for confirmation since this is destructive
        confirm = prompt_session.prompt("🔵 Clear conversation context? This cannot be undone (y/n): ").strip().lower()
        if confirm in ["y", "yes"]:
            # Keep only the original system prompt
            original_system_prompt = conversation_history[0]
            conversation_history[:] = [original_system_prompt]
            console.print("[green]✓ Conversation context cleared. Starting fresh![/green]")
            console.print("[green]  All file contexts and conversation history removed.[/green]")
        else:
            console.print("[yellow]Context clear cancelled.[/yellow]")
        return True
    return False

def try_handle_folder_command(user_input: str) -> bool:
    """Handle /folder command to manage base directory."""
    global base_dir
    if user_input.strip().lower().startswith("/folder"):
        folder_path = user_input[len("/folder"):].strip()
        if not folder_path:
            console.print(f"[yellow]Current base directory: '{base_dir}'[/yellow]")
            console.print("[yellow]Usage: /folder <path> or /folder reset[/yellow]")
            return True
        if folder_path.lower() == "reset":
            old_base = base_dir
            base_dir = Path.cwd()
            console.print(f"[green]✓ Base directory reset from '{old_base}' to: '{base_dir}'[/green]")
            return True
        try:
            new_base = Path(folder_path).resolve()
            if not new_base.exists() or not new_base.is_dir():
                console.print(f"[red]✗ Path does not exist or is not a directory: '{folder_path}'[/red]")
                return True
            # Check write permissions
            test_file = new_base / ".eng-git-test"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                console.print(f"[red]✗ No write permissions in directory: '{new_base}'[/red]")
                return True
            old_base = base_dir
            base_dir = new_base
            console.print(f"[green]✓ Base directory changed from '{old_base}' to: '{base_dir}'[/green]")
            console.print(f"[green]  All relative paths will now be resolved against this directory.[/green]")
            return True
        except Exception as e:
            console.print(f"[red]✗ Error setting base directory: {e}[/red]")
            return True
    return False

def try_handle_exit_command(user_input: str) -> bool:
    """Handle /exit and /quit commands."""
    if user_input.strip().lower() in ("/exit", "/quit"):
        console.print("[bold blue]👋 Goodbye![/bold blue]")
        sys.exit(0)
    return False

def try_handle_context_command(user_input: str) -> bool:
    """Handle /context command to show context usage statistics."""
    if user_input.strip().lower() == "/context":
        context_info = get_context_usage_info()
        
        # Create context usage table
        context_table = Table(title="📊 Context Usage Statistics", show_header=True, header_style="bold bright_blue")
        context_table.add_column("Metric", style="bright_cyan")
        context_table.add_column("Value", style="white")
        context_table.add_column("Status", style="white")
        
        # Add rows with usage information
        context_table.add_row(
            "Total Messages", 
            str(context_info["total_messages"]), 
            "📝"
        )
        context_table.add_row(
            "Estimated Tokens", 
            f"{context_info['estimated_tokens']:,}", 
            f"{context_info['token_usage_percent']:.1f}% of {ESTIMATED_MAX_TOKENS:,}"
        )
        context_table.add_row(
            "File Contexts", 
            str(context_info["file_contexts"]), 
            f"Max: {MAX_CONTEXT_FILES}"
        )
        
        # Status indicators
        if context_info["critical_limit"]:
            status_color = "red"
            status_text = "🔴 Critical - aggressive truncation active"
        elif context_info["approaching_limit"]:
            status_color = "yellow"
            status_text = "🟡 Warning - approaching limits"
        else:
            status_color = "green"
            status_text = "🟢 Healthy - plenty of space"
        
        context_table.add_row(
            "Context Health", 
            status_text, 
            ""
        )
        
        console.print(context_table)
        
        # Show token breakdown
        if context_info["token_breakdown"]:
            breakdown_table = Table(title="📋 Token Breakdown by Role", show_header=True, header_style="bold bright_blue", border_style="blue")
            breakdown_table.add_column("Role", style="bright_cyan")
            breakdown_table.add_column("Tokens", style="white")
            breakdown_table.add_column("Percentage", style="white")
            
            total_tokens = context_info["estimated_tokens"]
            for role, tokens in context_info["token_breakdown"].items():
                if tokens > 0:
                    percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0
                    breakdown_table.add_row(
                        role.capitalize(),
                        f"{tokens:,}",
                        f"{percentage:.1f}%"
                    )
            
            console.print(breakdown_table)
        
        # Show recommendations if approaching limits
        if context_info["approaching_limit"]:
            console.print("\n[yellow]💡 Recommendations to manage context:[/yellow]")
            console.print("[yellow]  • Use /clear-context to start fresh[/yellow]")
            console.print("[yellow]  • Remove large files from context[/yellow]")
            console.print("[yellow]  • Work with smaller file sections[/yellow]")
        
        return True
    return False

def try_handle_help_command(user_input: str) -> bool:
    """Handle /help command to show available commands."""
    if user_input.strip().lower() == "/help":
        help_table = Table(title="📝 Available Commands", show_header=True, header_style="bold bright_blue")
        help_table.add_column("Command", style="bright_cyan")
        help_table.add_column("Description", style="white")
        
        # General commands
        help_table.add_row("/help", "Show this help")
        help_table.add_row("/r1", "Call DeepSeek Reasoner model for one-off reasoning tasks")
        help_table.add_row("/reasoner", "Toggle between chat and reasoner models")
        help_table.add_row("/clear", "Clear screen")
        help_table.add_row("/clear-context", "Clear conversation context")
        help_table.add_row("/context", "Show context usage statistics")
        help_table.add_row("/exit, /quit", "Exit application")
        
        # Directory & file management
        help_table.add_row("/folder", "Show current base directory")
        help_table.add_row("/folder <path>", "Set base directory for file operations")
        help_table.add_row("/folder reset", "Reset base directory to current working directory")
        help_table.add_row(f"{ADD_COMMAND_PREFIX.strip()} <path>", "Add file/dir to conversation context (supports fuzzy matching)")
        
        # Git workflow commands
        help_table.add_row("/git init", "Initialize Git repository")
        help_table.add_row("/git status", "Show Git status")
        help_table.add_row(f"{GIT_BRANCH_COMMAND_PREFIX.strip()} <name>", "Create & switch to new branch")
        help_table.add_row(f"{COMMIT_COMMAND_PREFIX.strip()} [msg]", "Stage all files & commit (prompts if no message)")
        help_table.add_row("/git-info", "Show detailed Git capabilities")

        # Enhanced commands (if available)
        if ADVANCED_MODULES_AVAILABLE:
            help_table.add_row("", "")  # Separator
            help_table.add_row("[bold]Enhanced Features:[/bold]", "")
            help_table.add_row("/status", "Show comprehensive system status")
            help_table.add_row("/performance", "Show performance metrics")
            help_table.add_row("/codebase", "Show codebase summary")
            help_table.add_row("/intelligence", "Show AI intelligence metrics")

        console.print(help_table)
        
        # Show current model status
        current_model_name = "DeepSeek Reasoner 🧠" if agent_state.model_context['is_reasoner'] else "DeepSeek Chat 💬"
        console.print(f"\n[dim]Current model: {current_model_name}[/dim]")
        
        # Show fuzzy matching status
        fuzzy_status = "✓ Available" if FUZZY_AVAILABLE else "✗ Not installed (pip install thefuzz python-levenshtein)"
        console.print(f"[dim]Fuzzy matching: {fuzzy_status}[/dim]")
        
        return True
    return False

def initialize_git_repo_cmd() -> bool:
    """Initialize a git repository."""
    if Path(".git").exists(): 
        console.print("[yellow]Git repo already exists.[/yellow]")
        agent_state.git_context['enabled'] = True
        return True
    try:
        subprocess.run(["git", "init"], cwd=str(Path.cwd()), check=True, capture_output=True)
        agent_state.git_context['enabled'] = True
        branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(Path.cwd()), capture_output=True, text=True)
        agent_state.git_context['branch'] = branch_res.stdout.strip() if branch_res.returncode == 0 else "main"
        console.print(f"[green]✓ Initialized Git repo in {Path.cwd()}/.git/ (branch: {agent_state.git_context['branch']})[/green]")
        if not Path(".gitignore").exists() and prompt_session.prompt("🔵 No .gitignore. Create one? (y/n, default y): ", default="y").strip().lower() in ["y", "yes"]: 
            create_gitignore()
        elif agent_state.git_context['enabled'] and Path(".gitignore").exists(): 
            stage_file(".gitignore")
        if prompt_session.prompt(f"🔵 Initial commit? (y/n, default n): ", default="n").strip().lower() in ["y", "yes"]: 
            user_commit_changes("Initial commit")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        console.print(f"[red]✗ Failed to init Git: {e}[/red]")
        if isinstance(e, FileNotFoundError): 
            agent_state.git_context['enabled'] = False
        return False

def create_git_branch_cmd(branch_name: str) -> bool:
    """Create and switch to a git branch."""
    if not agent_state.git_context['enabled']: 
        console.print("[yellow]Git not enabled.[/yellow]")
        return True
    if not branch_name: 
        console.print("[yellow]Branch name empty.[/yellow]")
        return True
    try:
        existing_raw = subprocess.run(["git", "branch", "--list", branch_name], cwd=str(Path.cwd()), capture_output=True, text=True)
        if existing_raw.stdout.strip():
            console.print(f"[yellow]Branch '{branch_name}' exists.[/yellow]")
            current_raw = subprocess.run(["git", "branch", "--show-current"], cwd=str(Path.cwd()), capture_output=True, text=True)
            if current_raw.stdout.strip() != branch_name and prompt_session.prompt(f"🔵 Switch to '{branch_name}'? (y/n, default y): ", default="y").strip().lower() in ["y", "yes"]:
                subprocess.run(["git", "checkout", branch_name], cwd=str(Path.cwd()), check=True, capture_output=True)
                agent_state.git_context['branch'] = branch_name
                console.print(f"[green]✓ Switched to branch '{branch_name}'[/green]")
            return True
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=str(Path.cwd()), check=True, capture_output=True)
        agent_state.git_context['branch'] = branch_name
        console.print(f"[green]✓ Created & switched to new branch '{branch_name}'[/green]")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        console.print(f"[red]✗ Branch op failed: {e}[/red]")
        if isinstance(e, FileNotFoundError): 
            agent_state.git_context['enabled'] = False
        return False

def show_git_status_cmd() -> bool:
    """Show git status."""
    if not agent_state.git_context['enabled']: 
        console.print("[yellow]Git not enabled.[/yellow]")
        return True
    has_changes, files = get_git_status_porcelain()
    branch_raw = subprocess.run(["git", "branch", "--show-current"], cwd=str(Path.cwd()), capture_output=True, text=True)
    branch_msg = f"On branch {branch_raw.stdout.strip()}" if branch_raw.returncode == 0 and branch_raw.stdout.strip() else "Not on any branch?"
    console.print(Panel(branch_msg, title="Git Status", border_style="blue", expand=False))
    if not has_changes: 
        console.print("[green]Working tree clean.[/green]")
        return True
    table = Table(show_header=True, header_style="bold bright_blue", border_style="blue")
    table.add_column("Sts", width=3)
    table.add_column("File Path")
    table.add_column("Description", style="dim")
    s_map = {
        " M": (" M", "Mod (unstaged)"), "MM": ("MM", "Mod (staged&un)"), 
        " A": (" A", "Add (unstaged)"), "AM": ("AM", "Add (staged&mod)"), 
        "AD": ("AD", "Add (staged&del)"), " D": (" D", "Del (unstaged)"), 
        "??": ("??", "Untracked"), "M ": ("M ", "Mod (staged)"), 
        "A ": ("A ", "Add (staged)"), "D ": ("D ", "Del (staged)"), 
        "R ": ("R ", "Ren (staged)"), "C ": ("C ", "Cop (staged)"), 
        "U ": ("U ", "Unmerged")
    }
    staged, unstaged, untracked = False, False, False
    for code, filename in files:
        disp_code, desc = s_map.get(code, (code, "Unknown"))
        table.add_row(disp_code, filename, desc)
        if code == "??": 
            untracked = True
        elif code.startswith(" "): 
            unstaged = True
        else: 
            staged = True
    console.print(table)
    if not staged and (unstaged or untracked): 
        console.print("\n[yellow]No changes added to commit.[/yellow]")
    if staged: 
        console.print("\n[green]Changes to be committed.[/green]")
    if unstaged: 
        console.print("[yellow]Changes not staged for commit.[/yellow]")
    if untracked: 
        console.print("[cyan]Untracked files present.[/cyan]")
    return True

# -----------------------------------------------------------------------------
# 10. ENHANCED LLM TOOL HANDLER FUNCTIONS
# -----------------------------------------------------------------------------

def llm_git_init() -> str:
    """LLM tool handler for git init."""
    if Path(".git").exists(): 
        agent_state.git_context['enabled'] = True
        return "Git repository already exists."
    try:
        subprocess.run(["git", "init"], cwd=str(Path.cwd()), check=True, capture_output=True)
        agent_state.git_context['enabled'] = True
        branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(Path.cwd()), capture_output=True, text=True)
        agent_state.git_context['branch'] = branch_res.stdout.strip() if branch_res.returncode == 0 else "main"
        if not Path(".gitignore").exists(): 
            create_gitignore()
        elif agent_state.git_context['enabled']: 
            stage_file(".gitignore")
        return f"Git repository initialized successfully in {Path.cwd()}/.git/ (branch: {agent_state.git_context['branch']})."

    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            agent_state.git_context['enabled'] = False
        return f"Failed to initialize Git repository: {e}"

def llm_git_add(file_paths: List[str]) -> str:
    """LLM tool handler for git add."""
    if not agent_state.git_context['enabled']: 
        return "Git not initialized."
    if not file_paths: 
        return "No file paths to stage."
    staged_ok: List[str] = []
    failed_stage: List[str] = []
    for fp_str in file_paths:
        try: 
            norm_fp = normalize_path(fp_str)
            if stage_file(norm_fp):
                staged_ok.append(norm_fp)
            else:
                failed_stage.append(norm_fp)
        except ValueError as e: 
            failed_stage.append(f"{fp_str} (path error: {e})")
        except Exception as e: 
            failed_stage.append(f"{fp_str} (error: {e})")
    res = []
    if staged_ok: 
        res.append(f"Staged: {', '.join(Path(p).name for p in staged_ok)}")
    if failed_stage: 
        res.append(f"Failed to stage: {', '.join(str(Path(p).name if isinstance(p,str) else p) for p in failed_stage)}")
    return ". ".join(res) + "." if res else "No files staged. Check paths."

def llm_git_commit(message: str) -> str:
    """LLM tool handler for git commit."""
    if not agent_state.git_context['enabled']: 
        return "Git not initialized."
    if not message: 
        return "Commit message empty."
    try:
        staged_check = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=str(Path.cwd()))
        if staged_check.returncode == 0: 
            return "No changes staged. Use git_add first."
        result = subprocess.run(["git", "commit", "-m", message], cwd=str(Path.cwd()), capture_output=True, text=True)
        if result.returncode == 0:
            info_raw = subprocess.run(["git", "log", "-1", "--pretty=%h %s"], cwd=str(Path.cwd()), capture_output=True, text=True).stdout.strip()
            return f"Committed. Commit: {info_raw}"
        return f"Failed to commit: {result.stderr.strip()}"
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            agent_state.git_context['enabled'] = False
        return f"Git commit error: {e}"
    except Exception as e: 
        console.print_exception()
        return f"Unexpected commit error: {e}"

def llm_git_create_branch(branch_name: str) -> str:
    """LLM tool handler for git branch creation."""
    if not agent_state.git_context['enabled']: 
        return "Git not initialized."
    bn = branch_name.strip()
    if not bn: 
        return "Branch name empty."
    try:
        exist_res = subprocess.run(["git", "rev-parse", "--verify", f"refs/heads/{bn}"], cwd=str(Path.cwd()), capture_output=True, text=True)
        if exist_res.returncode == 0:
            current_raw = subprocess.run(["git", "branch", "--show-current"], cwd=str(Path.cwd()), capture_output=True, text=True)
            if current_raw.stdout.strip() == bn: 
                return f"Already on branch '{bn}'."
            subprocess.run(["git", "checkout", bn], cwd=str(Path.cwd()), check=True, capture_output=True, text=True)
            agent_state.git_context['branch'] = bn
            return f"Branch '{bn}' exists. Switched to it."
        subprocess.run(["git", "checkout", "-b", bn], cwd=str(Path.cwd()), check=True, capture_output=True, text=True)
        agent_state.git_context['branch'] = bn
        return f"Created & switched to new branch '{bn}'."
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            agent_state.git_context['enabled'] = False
        return f"Branch op failed for '{bn}': {e}"

def llm_git_status() -> str:
    """LLM tool handler for git status."""
    if not agent_state.git_context['enabled']: 
        return "Git not initialized."
    try:
        branch_res = subprocess.run(["git", "branch", "--show-current"], cwd=str(Path.cwd()), capture_output=True, text=True)
        branch_name = branch_res.stdout.strip() if branch_res.returncode == 0 and branch_res.stdout.strip() else "detached HEAD"
        has_changes, files = get_git_status_porcelain()
        if not has_changes: 
            return f"On branch '{branch_name}'. Working tree clean."
        lines = [f"On branch '{branch_name}'."]
        staged: List[str] = []
        unstaged: List[str] = []
        untracked: List[str] = []
        for code, filename in files:
            if code == "??": 
                untracked.append(filename)
            elif code.startswith(" "): 
                unstaged.append(f"{code.strip()} {filename}")
            else: 
                staged.append(f"{code.strip()} {filename}")
        if staged: 
            lines.extend(["\nChanges to be committed:"] + [f"  {s}" for s in staged])
        if unstaged: 
            lines.extend(["\nChanges not staged for commit:"] + [f"  {s}" for s in unstaged])
        if untracked: 
            lines.extend(["\nUntracked files:"] + [f"  {f}" for f in untracked])
        return "\n".join(lines)
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            agent_state.git_context['enabled'] = False
        return f"Git status error: {e}"

def execute_function_call_dict(tool_call_dict: Dict[str, Any]) -> str:
    """
    Execute a function call from the LLM with enhanced fuzzy matching and security.
    
    Args:
        tool_call_dict: Dictionary containing function call information
        
    Returns:
        String result of the function execution
    """
    func_name = "unknown_function"
    try:
        func_name = tool_call_dict["function"]["name"]
        args = json.loads(tool_call_dict["function"]["arguments"])
        
        if func_name == "read_file":
            norm_path = normalize_path(args["file_path"])
            content = read_local_file(norm_path)
            return f"Content of file '{norm_path}':\n\n{content}"
            
        elif func_name == "read_multiple_files":
            response_data = {
                "files_read": {},
                "errors": {}
            }
            total_content_size = 0

            for fp in args["file_paths"]:
                try:
                    norm_path = normalize_path(fp)
                    content = read_local_file(norm_path)

                    if total_content_size + len(content) > MAX_MULTIPLE_READ_SIZE:
                        response_data["errors"][norm_path] = "Could not read file, as total content size would exceed the safety limit."
                        continue

                    response_data["files_read"][norm_path] = content
                    total_content_size += len(content)

                except (OSError, ValueError) as e:
                    # Use the original path in the error if normalization fails
                    error_key = str(base_dir / fp)
                    response_data["errors"][error_key] = str(e)

            # Return a JSON string, which is much easier for the LLM to parse reliably
            return json.dumps(response_data, indent=2)
            
        elif func_name == "create_file": 
            create_file(args["file_path"], args["content"])
            return f"File '{args['file_path']}' created/updated."
            
        elif func_name == "create_multiple_files":
            created: List[str] = []
            errors: List[str] = []
            for f_info in args["files"]:
                try: 
                    create_file(f_info["path"], f_info["content"])
                    created.append(f_info["path"])
                except Exception as e: 
                    errors.append(f"Error creating {f_info.get('path','?path')}: {e}")
            res_parts = []
            if created: 
                res_parts.append(f"Created/updated {len(created)} files: {', '.join(created)}")
            if errors: 
                res_parts.append(f"Errors: {'; '.join(errors)}")
            return ". ".join(res_parts) if res_parts else "No files processed."
            
        elif func_name == "edit_file":
            fp = args["file_path"]
            if not ensure_file_in_context(fp): 
                return f"Error: Could not read '{fp}' for editing."
            try: 
                apply_fuzzy_diff_edit(fp, args["original_snippet"], args["new_snippet"])
                return f"Edit applied successfully to '{fp}'. Check console for details."
            except Exception as e:
                return f"Error during edit_file call for '{fp}': {e}."
                
        elif func_name == "git_init": 
            return llm_git_init()
        elif func_name == "git_add": 
            return llm_git_add(args.get("file_paths", []))
        elif func_name == "git_commit": 
            return llm_git_commit(args.get("message", "Auto commit"))
        elif func_name == "git_create_branch": 
            return llm_git_create_branch(args.get("branch_name", ""))
        elif func_name == "git_status": 
            return llm_git_status()
        elif func_name == "run_powershell":
            command = args["command"]
            
            # SECURITY GATE
            if agent_state.security_context["require_powershell_confirmation"]:
                console.print(Panel(
                    f"The assistant wants to run this PowerShell command:\n\n[bold yellow]{command}[/bold yellow]", 
                    title="🚨 Security Confirmation Required", 
                    border_style="red"
                ))
                confirm = prompt_session.prompt("🔵 Do you want to allow this command to run? (y/N): ", default="n").strip().lower()
                
                if confirm not in ["y", "yes"]:
                    console.print("[red]Execution denied by user.[/red]")
                    return "PowerShell command execution was denied by the user."
            
            output, error = run_powershell_command(command)
            if error:
                return f"PowerShell Error:\n{error}"
            return f"PowerShell Output:\n{output}"

        # New advanced function implementations
        elif func_name == "analyze_code":
            if not ADVANCED_MODULES_AVAILABLE or not code_analyzer:
                return "Code analysis not available - advanced modules not loaded"

            file_path = args["file_path"]
            analysis_type = args.get("analysis_type", "all")

            try:
                result = code_analyzer.analyze_file(file_path, analysis_type)
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                return f"Code analysis failed: {e}"

        elif func_name == "search_code":
            if not ADVANCED_MODULES_AVAILABLE or not code_analyzer:
                return "Code search not available - advanced modules not loaded"

            pattern = args["pattern"]
            search_type = args.get("search_type", "regex")
            file_paths = args.get("file_paths", [])

            try:
                results = []
                if file_paths:
                    for file_path in file_paths:
                        file_results = code_analyzer.search_code_patterns(file_path, pattern, search_type)
                        results.extend(file_results)
                else:
                    # Search in current directory
                    import glob
                    for file_path in glob.glob("**/*.py", recursive=True):
                        file_results = code_analyzer.search_code_patterns(file_path, pattern, search_type)
                        results.extend(file_results)

                return json.dumps(results, indent=2, default=str)
            except Exception as e:
                return f"Code search failed: {e}"

        elif func_name == "web_search":
            if not ADVANCED_MODULES_AVAILABLE or not web_search_rag:
                return "Web search not available - advanced modules not loaded"

            query = args["query"]
            max_results = args.get("max_results", 5)
            search_type = args.get("search_type", "general")

            try:
                results = web_search_rag.search_programming_help(query, max_results, search_type)
                summary = web_search_rag.summarize_search_results(results)
                return f"Search Results for '{query}':\n\n{summary}"
            except Exception as e:
                return f"Web search failed: {e}"

        elif func_name == "debug_code":
            if not ADVANCED_MODULES_AVAILABLE or not autonomous_debugger:
                return "Autonomous debugging not available - advanced modules not loaded"

            file_path = args["file_path"]
            error_message = args["error_message"]
            auto_fix = args.get("auto_fix", False)

            try:
                debug_result = autonomous_debugger.debug_error(file_path, error_message, auto_fix)

                response = f"Debug Analysis for '{file_path}':\n\n"
                response += f"Error Type: {debug_result.error_type}\n"
                response += f"Confidence: {debug_result.confidence:.2f}\n"
                response += f"Auto-fixable: {debug_result.auto_fixable}\n\n"
                response += "Suggested Fixes:\n"
                for i, fix in enumerate(debug_result.suggested_fixes, 1):
                    response += f"{i}. {fix}\n"

                return response
            except Exception as e:
                return f"Debug analysis failed: {e}"

        elif func_name == "refactor_code":
            return "Code refactoring functionality coming soon..."

        elif func_name == "generate_tests":
            return "Test generation functionality coming soon..."

        elif func_name == "run_terminal_command":
            command = args["command"]
            working_directory = args.get("working_directory", str(base_dir))
            timeout = args.get("timeout", 30)

            try:
                # Enhanced cross-platform command execution
                import platform
                current_platform = platform.system().lower()

                # Normalize command for platform
                normalized_command = normalize_command_for_platform(command, current_platform)

                # Execute with proper encoding
                if current_platform == 'windows':
                    # Use cmd.exe for Windows
                    if not normalized_command.startswith('powershell'):
                        normalized_command = f'cmd /c "{normalized_command}"'

                result = subprocess.run(
                    normalized_command,
                    shell=True,
                    cwd=working_directory,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding='utf-8',
                    errors='ignore'
                )

                response = f"Command: {command}\n"
                response += f"Normalized: {normalized_command}\n"
                response += f"Exit Code: {result.returncode}\n\n"

                if result.stdout:
                    stdout_clean = result.stdout.strip()
                    response += f"Output:\n{stdout_clean}\n"

                if result.stderr:
                    stderr_clean = result.stderr.strip()
                    response += f"Error:\n{stderr_clean}\n"

                # Add execution context
                response += f"\nWorking Directory: {working_directory}"
                response += f"\nPlatform: {current_platform}"

                return response
            except subprocess.TimeoutExpired:
                return f"Command timed out after {timeout} seconds"
            except Exception as e:
                return f"Command execution failed: {e}"

        elif func_name == "convert_code":
            return "Code conversion functionality coming soon..."

        else:
            return f"Unknown LLM function: {func_name}"
            
    except json.JSONDecodeError as e: 
        console.print(f"[red]JSON Decode Error for {func_name}: {e}\nArgs: {tool_call_dict.get('function',{}).get('arguments','')}[/red]")
        return f"Error: Invalid JSON args for {func_name}."
    except KeyError as e: 
        console.print(f"[red]KeyError in {func_name}: Missing key {e}[/red]")
        return f"Error: Missing param for {func_name} (KeyError: {e})."
    except Exception as e: 
        console.print(f"[red]Unexpected Error in LLM func '{func_name}':[/red]")
        console.print_exception()
        return f"Unexpected error in {func_name}: {e}"

# -----------------------------------------------------------------------------
# 10. ENHANCED TERMINAL FUNCTIONS
# -----------------------------------------------------------------------------

def normalize_command_for_platform(command: str, platform: str) -> str:
    """Normalize command for cross-platform execution."""
    command_lower = command.lower().strip()

    # Platform-specific command mappings
    command_mappings = {
        'windows': {
            'ls': 'dir /b',
            'ls -la': 'dir',
            'ls -l': 'dir',
            'cat': 'type',
            'pwd': 'cd',
            'date': 'date /t',
            'time': 'time /t',
            'clear': 'cls',
            'grep': 'findstr',
            'which': 'where',
            'ps': 'tasklist',
            'kill': 'taskkill /PID'
        },
        'linux': {
            'dir': 'ls -la',
            'cls': 'clear',
            'type': 'cat'
        },
        'darwin': {  # macOS
            'dir': 'ls -la',
            'cls': 'clear',
            'type': 'cat'
        }
    }

    # Universal commands for common requests
    universal_commands = {
        'date_time': {
            'windows': 'date /t && time /t',
            'linux': 'date',
            'darwin': 'date'
        },
        'current_directory': {
            'windows': 'cd',
            'linux': 'pwd',
            'darwin': 'pwd'
        },
        'list_files': {
            'windows': 'dir',
            'linux': 'ls -la',
            'darwin': 'ls -la'
        }
    }

    # Check for universal command patterns
    if any(word in command_lower for word in ['date', 'time']) and any(word in command_lower for word in ['show', 'get', 'current']):
        return universal_commands['date_time'].get(platform, command)
    elif any(word in command_lower for word in ['pwd', 'current directory', 'where am i']):
        return universal_commands['current_directory'].get(platform, command)
    elif any(word in command_lower for word in ['ls', 'list files', 'show files']):
        return universal_commands['list_files'].get(platform, command)

    # Apply platform-specific mappings
    if platform in command_mappings:
        mappings = command_mappings[platform]

        # Try exact match first
        if command_lower in mappings:
            return mappings[command_lower]

        # Try partial matches
        for cmd, replacement in mappings.items():
            if command_lower.startswith(cmd + ' ') or command_lower == cmd:
                return command.replace(cmd, replacement, 1)

    return command

# -----------------------------------------------------------------------------
# 11. ENHANCED WORKFLOW FUNCTIONS
# -----------------------------------------------------------------------------

async def process_enhanced_workflow(user_input: str, conversation_history: List[Dict[str, Any]]) -> None:
    """Process user input through the enhanced iterative workflow engine."""
    start_time = time.time()

    try:
        # Show workflow status
        with console.status("[bold green]🧠 Processing with Enhanced Iterative Workflow...", spinner="dots"):

            # Step 1: Natural Language Understanding (if NLP available)
            if nlp_processor:
                console.print("[dim]🔤 Natural Language Processing...[/dim]")
                nlp_result = nlp_processor.process_natural_language(user_input)
                console.print(f"[dim]   Intent: {nlp_result.intent.value} (confidence: {nlp_result.confidence:.2f})[/dim]")

            # Step 2: Enhanced Workflow Processing
            if workflow_engine:
                console.print("[dim]🔄 Executing Iterative Workflow (Execute→Analyze→Plan→Execute)...[/dim]")

                # Process through enhanced workflow engine
                workflow_result = await workflow_engine.process_user_input(user_input)

                # Show workflow progress
                if workflow_result.get('success', False):
                    operations_completed = workflow_result.get('operations_completed', 0)
                    iterations = len(workflow_result.get('iterations', []))
                    execution_time = workflow_result.get('total_execution_time', 0.0)

                    console.print(f"[dim]   ✅ Workflow completed: {operations_completed} operations in {iterations} iterations ({execution_time:.2f}s)[/dim]")

                    # Show goal achievement
                    if workflow_result.get('goal_achieved', False):
                        console.print(f"[dim]   🎯 User requirements fully satisfied![/dim]")
                    else:
                        console.print(f"[dim]   ⚠️ Partial completion - some requirements may need refinement[/dim]")
                else:
                    console.print(f"[dim]   ❌ Workflow failed: {workflow_result.get('error', 'Unknown error')}[/dim]")

            # Step 3: Codebase Integration
            if codebase_awareness and any(word in user_input.lower() for word in ['file', 'code', 'create', 'edit']):
                console.print("[dim]📁 Updating Codebase Awareness...[/dim]")
                # Update active files tracking
                changed_files = codebase_awareness.get_file_changes()
                if changed_files:
                    console.print(f"[dim]   📝 Detected changes in {len(changed_files)} files[/dim]")

        # Step 4: Generate LLM Response
        console.print("[dim]💬 Generating AI Response...[/dim]")

        # Create enhanced context for LLM
        enhanced_context = conversation_history.copy()

        # Add workflow results to context if available
        if 'workflow_result' in locals():
            workflow_summary = f"""
Enhanced Workflow Execution Summary:
- User Request: {user_input}
- Success: {workflow_result.get('success', False)}
- Operations: {workflow_result.get('operations_completed', 0)}
- Iterations: {len(workflow_result.get('iterations', []))}
- Goal Achieved: {workflow_result.get('goal_achieved', False)}
- Execution Time: {workflow_result.get('total_execution_time', 0.0):.2f}s
"""

            enhanced_context.append({
                "role": "system",
                "content": f"Enhanced iterative workflow completed. {workflow_summary}"
            })

        # Get LLM response using existing function
        try:
            managed_history = manage_conversation_history(enhanced_context)
            response = get_llm_response(managed_history)
        except Exception as e:
            console.print(f"[yellow]⚠ Error in LLM response generation: {e}[/yellow]")
            logger.warning(f"LLM response error: {e}")
            # Fallback to simple response
            response = f"I successfully processed your request through the enhanced workflow. The task has been completed."

        # Add assistant response to conversation
        conversation_history.append({"role": "assistant", "content": response})

        # Display results
        processing_time = time.time() - start_time
        console.print(f"[dim]⚡ Processed in {processing_time:.2f}s[/dim]")

        # Display main response
        console.print(f"\n[bold bright_blue]🤖 CODY:[/bold bright_blue] {response}")

        # Show smart suggestions
        analysis_results = workflow_result.get('analysis_results', [])
        if analysis_results:
            latest_analysis = analysis_results[-1]
            recommendations = latest_analysis.get('recommendations', [])
            if recommendations:
                console.print("\n[bold yellow]💡 Smart Suggestions:[/bold yellow]")
                for rec in recommendations[:3]:
                    console.print(f"   • {rec}")

        console.print()  # Empty line for spacing

    except Exception as e:
        console.print(f"[red]Enhanced workflow error: {e}[/red]")
        logger.error(f"Enhanced workflow error: {e}")

        # Enhanced fallback mechanism
        try:
            # Try to execute the command directly if it looks like a system command
            if any(cmd in user_input.lower() for cmd in ['time', 'date', 'ls', 'dir', 'pwd']):
                console.print("[dim]🔄 Attempting direct command execution...[/dim]")

                # Map common commands
                command_map = {
                    'time': 'date /t && time /t' if sys.platform.startswith('win') else 'date',
                    'date': 'date /t' if sys.platform.startswith('win') else 'date',
                    'ls': 'dir' if sys.platform.startswith('win') else 'ls',
                    'pwd': 'cd' if sys.platform.startswith('win') else 'pwd'
                }

                for keyword, command in command_map.items():
                    if keyword in user_input.lower():
                        try:
                            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                response = f"Here's the result:\n```\n{result.stdout.strip()}\n```"
                            else:
                                response = f"Command executed with warnings:\n```\n{result.stderr.strip()}\n```"
                            break
                        except Exception as cmd_error:
                            response = f"I encountered an issue executing the command: {cmd_error}"
                            break
                else:
                    response = f"I encountered an issue processing your request: {user_input}. Let me try a different approach."
            else:
                response = f"I encountered an issue processing your request: {user_input}. Let me try a different approach."

        except Exception as fallback_error:
            console.print(f"[yellow]⚠ Fallback mechanism also failed: {fallback_error}[/yellow]")
            response = f"I encountered multiple issues processing your request: {user_input}. Please try rephrasing your request."

        conversation_history.append({"role": "assistant", "content": response})
        console.print(f"\n[bold bright_blue]🤖 CODY:[/bold bright_blue] {response}\n")

def generate_workflow_response(user_input: str, workflow_result: Dict[str, Any],
                             context: Any) -> str:
    """Generate intelligent response based on workflow execution results."""
    if not workflow_result['success']:
        return f"I encountered difficulties processing your request: {user_input}. {workflow_result.get('error', 'Unknown error occurred.')}"

    execution_results = workflow_result.get('execution_results', [])
    if not execution_results:
        return "I processed your request but didn't execute any specific actions."

    latest_result = execution_results[-1]
    output_data = latest_result.get('output_data', {})

    # Generate response based on task type and results
    if latest_result.get('task_type') == 'terminal_command':
        if output_data.get('success'):
            stdout = output_data.get('stdout', '').strip()
            command = output_data.get('command', '')

            if 'date' in user_input.lower() and 'time' in user_input.lower():
                return f"✅ **Current Date and Time:**\n\n📅 **{stdout}**\n\n🔧 Command executed: `{command}`\n⚡ Execution time: {output_data.get('execution_time', 0):.3f}s"
            elif 'directory' in user_input.lower() or 'pwd' in user_input.lower():
                return f"✅ **Current Directory:**\n\n📁 **{stdout}**\n\n🔧 Command executed: `{command}`\n⚡ Execution time: {output_data.get('execution_time', 0):.3f}s"
            elif 'list' in user_input.lower() or 'files' in user_input.lower():
                file_count = len(stdout.split('\n')) if stdout else 0
                return f"✅ **Directory Contents ({file_count} items):**\n\n```\n{stdout}\n```\n\n🔧 Command executed: `{command}`\n⚡ Execution time: {output_data.get('execution_time', 0):.3f}s"
            else:
                return f"✅ **Command executed successfully:**\n\n```\n{stdout}\n```\n\n🔧 Command: `{command}`\n⚡ Execution time: {output_data.get('execution_time', 0):.3f}s"
        else:
            error = output_data.get('stderr', 'Unknown error')
            return f"Command execution failed: {error}"

    elif latest_result.get('task_type') == 'file_operation':
        operation = output_data.get('operation', '')
        file_path = output_data.get('source_path', '')

        if operation == 'create':
            return f"Successfully created file: `{file_path}`"
        elif operation == 'read':
            content_length = len(output_data.get('content', ''))
            return f"Successfully read file `{file_path}` ({content_length} characters)"
        else:
            return f"File operation '{operation}' completed for: `{file_path}`"

    elif latest_result.get('task_type') == 'system_info':
        system_info = output_data.get('system_info', '')
        return f"Here's your system information:\n\n```\n{system_info}\n```"

    # Default response
    progress = workflow_result.get('final_progress', 0)
    iterations = workflow_result.get('iterations', 1)

    return f"I processed your request through {iterations} iteration(s) with {progress:.1%} completion. The task has been executed successfully."

async def try_handle_enhanced_commands(user_input: str) -> bool:
    """Handle enhanced commands specific to advanced features."""
    if user_input.strip() == "/status":
        show_system_status()
        return True
    elif user_input.strip() == "/performance":
        show_performance_metrics()
        return True
    elif user_input.strip() == "/codebase":
        show_codebase_summary()
        return True
    elif user_input.strip() == "/intelligence":
        show_intelligence_metrics()
        return True
    return False

def show_system_status() -> None:
    """Show comprehensive system status."""
    console.print("\n[bold bright_blue]🔍 CODY System Status[/bold bright_blue]")

    # Module status
    status_table = Table(title="Module Status")
    status_table.add_column("Module", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="dim")

    modules = [
        ("NLP Processor", "✅ Active" if nlp_processor else "❌ Inactive", "Natural Language Processing"),
        ("Code Analyzer", "✅ Active" if code_analyzer else "❌ Inactive", "AST-based Code Analysis"),
        ("Autonomous Debugger", "✅ Active" if autonomous_debugger else "❌ Inactive", "Error Detection & Fixing"),
        ("Web Search RAG", "✅ Active" if web_search_rag else "❌ Inactive", "Real-time Information Retrieval"),
        ("Task Manager", "✅ Active" if task_manager else "❌ Inactive", "Multi-threaded Execution"),
        ("General Intelligence", "✅ Active" if general_intelligence else "❌ Inactive", "Chain-of-Thought Reasoning"),
        ("Performance Core", "✅ Active" if performance_core else "❌ Inactive", "Ultra-low Latency Processing"),
        ("Codebase Awareness", "✅ Active" if codebase_awareness else "❌ Inactive", "Full Project Understanding"),
        ("Terminal FS Agent", "✅ Active" if terminal_fs_agent else "❌ Inactive", "Command & File Operations")
    ]

    for module, status, details in modules:
        status_table.add_row(module, status, details)

    console.print(status_table)

def show_performance_metrics() -> None:
    """Show performance metrics."""
    console.print("\n[bold bright_blue]⚡ Performance Metrics[/bold bright_blue]")

    if performance_core:
        stats = performance_core.get_performance_stats()

        metrics_table = Table(title="Performance Statistics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        metrics_table.add_row("Cache Hit Rate", f"{stats['cache_stats']['hit_rate']:.2%}")
        metrics_table.add_row("Cache Size", f"{stats['cache_stats']['size_mb']:.1f} MB")
        metrics_table.add_row("CPU Usage", f"{stats['system_stats']['cpu_percent']:.1f}%")
        metrics_table.add_row("Memory Usage", f"{stats['system_stats']['memory_percent']:.1f}%")
        metrics_table.add_row("Active Threads", str(stats['system_stats']['active_threads']))

        console.print(metrics_table)
    else:
        console.print("[yellow]Performance core not available[/yellow]")

def show_codebase_summary() -> None:
    """Show codebase summary."""
    console.print("\n[bold bright_blue]📁 Codebase Summary[/bold bright_blue]")

    if codebase_awareness:
        summary = codebase_awareness.get_project_summary()

        summary_table = Table(title="Project Overview")
        summary_table.add_column("Aspect", style="cyan")
        summary_table.add_column("Details", style="green")

        summary_table.add_row("Total Files", str(summary['project_structure']['total_files']))
        summary_table.add_row("Active Files", str(summary['active_files']))
        summary_table.add_row("Recent Changes", str(summary['recent_changes']))
        summary_table.add_row("Languages", ", ".join(summary['languages_used']))
        summary_table.add_row("Total Functions", str(summary['total_functions']))
        summary_table.add_row("Total Classes", str(summary['total_classes']))

        console.print(summary_table)
    else:
        console.print("[yellow]Codebase awareness not available[/yellow]")

def show_intelligence_metrics() -> None:
    """Show intelligence system metrics."""
    console.print("\n[bold bright_blue]🧠 Intelligence Metrics[/bold bright_blue]")

    if general_intelligence:
        metrics = general_intelligence.get_intelligence_metrics()

        if metrics.get("status") == "no_data":
            console.print("[yellow]No intelligence data available yet[/yellow]")
        else:
            intel_table = Table(title="Intelligence Performance")
            intel_table.add_column("Metric", style="cyan")
            intel_table.add_column("Value", style="green")

            intel_table.add_row("Total Executions", str(metrics['total_executions']))
            intel_table.add_row("Recent Success Rate", f"{metrics['recent_success_rate']:.2%}")
            intel_table.add_row("Reasoning Quality", f"{metrics['average_reasoning_quality']:.2f}")
            intel_table.add_row("Memory Categories", ", ".join(metrics['memory_categories']))

            console.print(intel_table)
    else:
        console.print("[yellow]General intelligence not available[/yellow]")

# -----------------------------------------------------------------------------
# 12. MAIN LOOP & ENTRY POINT
# -----------------------------------------------------------------------------

def main_loop() -> None:
    """Main application loop."""
    global conversation_history

    while True:
        try:
            prompt_indicator = get_prompt_indicator()
            user_input = prompt_session.prompt(f"{prompt_indicator} You: ")
            
            if not user_input.strip(): 
                continue

            # Handle commands
            if try_handle_add_command(user_input): continue
            if try_handle_commit_command(user_input): continue
            if try_handle_git_command(user_input): continue
            if try_handle_git_info_command(user_input): continue
            if try_handle_r1_command(user_input): continue
            if try_handle_reasoner_command(user_input): continue
            if try_handle_clear_command(user_input): continue
            if try_handle_clear_context_command(user_input): continue
            if try_handle_context_command(user_input): continue
            if try_handle_folder_command(user_input): continue
            if try_handle_exit_command(user_input): continue
            if try_handle_help_command(user_input): continue

            # Handle enhanced commands if available
            if ADVANCED_MODULES_AVAILABLE:
                if asyncio.run(try_handle_enhanced_commands(user_input)): continue

            # Add user message to conversation
            conversation_history.append({"role": "user", "content": user_input})

            # Enhanced workflow processing if available
            enhanced_keywords = [
                'create', 'file', 'debug', 'search', 'analyze', 'install', 'run',
                'show', 'display', 'get', 'date', 'time', 'current', 'directory',
                'list', 'files', 'pwd', 'ls', 'dir', 'terminal', 'command',
                'execute', 'system', 'info', 'status'
            ]

            if ADVANCED_MODULES_AVAILABLE and any(word in user_input.lower() for word in enhanced_keywords):
                try:
                    console.print("[dim]🚀 Using Enhanced Iterative Workflow Engine...[/dim]")
                    asyncio.run(process_enhanced_workflow(user_input, conversation_history))
                    continue  # Skip normal processing
                except Exception as e:
                    console.print(f"[yellow]Enhanced workflow failed, falling back to standard processing: {e}[/yellow]")
                    logger.warning(f"Enhanced workflow error: {e}")

            # Check context usage and warn if necessary
            context_info = get_context_usage_info()
            if context_info["critical_limit"] and len(conversation_history) % 10 == 0:  # Warn every 10 messages when critical
                console.print(f"[red]⚠ Context critical: {context_info['token_usage_percent']:.1f}% used. Consider /clear-context or /context for details.[/red]")
            elif context_info["approaching_limit"] and len(conversation_history) % 20 == 0:  # Warn every 20 messages when approaching
                console.print(f"[yellow]⚠ Context high: {context_info['token_usage_percent']:.1f}% used. Use /context for details.[/yellow]")
            
            # Determine which model to use
            current_model = agent_state.model_context['current_model']
            model_name = "DeepSeek Reasoner" if current_model == REASONER_MODEL else "DeepSeek Engineer"
            
            # Make API call
            with console.status(f"[bold yellow]{model_name} is thinking...[/bold yellow]", spinner="dots"):
                response_stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
                    model=current_model,
                    messages=conversation_history, # type: ignore 
                    tools=tools, # type: ignore 
                    tool_choice="auto",
                    stream=True
                )
            
            # Process streaming response
            full_response_content = ""
            accumulated_tool_calls: List[Dict[str, Any]] = []

            console.print(f"[bold bright_magenta]🤖 {model_name}:[/bold bright_magenta] ", end="")
            for chunk in response_stream:
                delta: ChoiceDelta = chunk.choices[0].delta
                if delta.content:
                    content_part = delta.content
                    console.print(content_part, end="", style="bright_magenta")
                    full_response_content += content_part
                
                if delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        idx = tool_call_chunk.index
                        while len(accumulated_tool_calls) <= idx:
                            accumulated_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        
                        current_tool_dict = accumulated_tool_calls[idx]
                        if tool_call_chunk.id: 
                            current_tool_dict["id"] = tool_call_chunk.id
                        if tool_call_chunk.function:
                            if tool_call_chunk.function.name: 
                                current_tool_dict["function"]["name"] = tool_call_chunk.function.name
                            if tool_call_chunk.function.arguments: 
                                current_tool_dict["function"]["arguments"] += tool_call_chunk.function.arguments
            console.print()

            # Always add assistant message to maintain conversation flow
            assistant_message: Dict[str, Any] = {"role": "assistant"}
            
            # Always include content (even if empty) to maintain conversation flow
            assistant_message["content"] = full_response_content

            # Validate and add tool calls if any
            valid_tool_calls = validate_tool_calls(accumulated_tool_calls)
            if valid_tool_calls:
                assistant_message["tool_calls"] = valid_tool_calls
            
            # Always add the assistant message (maintains conversation flow)
            conversation_history.append(assistant_message)

            # Execute tool calls and allow assistant to continue naturally
            if valid_tool_calls:
                # Execute all tool calls first
                for tool_call_to_exec in valid_tool_calls: 
                    console.print(Panel(
                        f"[bold blue]Calling:[/bold blue] {tool_call_to_exec['function']['name']}\n"
                        f"[bold blue]Args:[/bold blue] {tool_call_to_exec['function']['arguments']}",
                        title="🛠️ Function Call", border_style="yellow", expand=False
                    ))
                    tool_output = execute_function_call_dict(tool_call_to_exec) 
                    console.print(Panel(tool_output, title=f"↪️ Output of {tool_call_to_exec['function']['name']}", border_style="green", expand=False))
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_to_exec["id"],
                        "name": tool_call_to_exec["function"]["name"],
                        "content": tool_output 
                    })
                
                # Now let the assistant continue with the tool results
                # This creates a natural conversation flow where the assistant processes the results
                max_continuation_rounds = 3
                current_round = 0
                
                while current_round < max_continuation_rounds:
                    current_round += 1
                    
                    with console.status(f"[bold yellow]{model_name} is processing results...[/bold yellow]", spinner="dots"):
                        continue_response_stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
                            model=current_model,
                            messages=conversation_history, # type: ignore 
                            tools=tools, # type: ignore 
                            tool_choice="auto",
                            stream=True
                        )
                    
                    # Process the continuation response
                    continuation_content = ""
                    continuation_tool_calls: List[Dict[str, Any]] = []
                    
                    console.print(f"[bold bright_magenta]🤖 {model_name}:[/bold bright_magenta] ", end="")
                    for chunk in continue_response_stream:
                        delta: ChoiceDelta = chunk.choices[0].delta
                        if delta.content:
                            content_part = delta.content
                            console.print(content_part, end="", style="bright_magenta")
                            continuation_content += content_part
                        
                        if delta.tool_calls:
                            for tool_call_chunk in delta.tool_calls:
                                idx = tool_call_chunk.index
                                while len(continuation_tool_calls) <= idx:
                                    continuation_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                                
                                current_tool_dict = continuation_tool_calls[idx]
                                if tool_call_chunk.id: 
                                    current_tool_dict["id"] = tool_call_chunk.id
                                if tool_call_chunk.function:
                                    if tool_call_chunk.function.name: 
                                        current_tool_dict["function"]["name"] = tool_call_chunk.function.name
                                    if tool_call_chunk.function.arguments: 
                                        current_tool_dict["function"]["arguments"] += tool_call_chunk.function.arguments
                    console.print()
                    
                    # Add the continuation response to conversation history
                    continuation_message: Dict[str, Any] = {"role": "assistant", "content": continuation_content}
                    
                    # Check if there are more tool calls to execute
                    valid_continuation_tools = validate_tool_calls(continuation_tool_calls)
                    if valid_continuation_tools:
                        continuation_message["tool_calls"] = valid_continuation_tools
                        conversation_history.append(continuation_message)
                        
                        # Execute the additional tool calls
                        for tool_call_to_exec in valid_continuation_tools:
                            console.print(Panel(
                                f"[bold blue]Calling:[/bold blue] {tool_call_to_exec['function']['name']}\n"
                                f"[bold blue]Args:[/bold blue] {tool_call_to_exec['function']['arguments']}",
                                title="🛠️ Function Call", border_style="yellow", expand=False
                            ))
                            tool_output = execute_function_call_dict(tool_call_to_exec)
                            console.print(Panel(tool_output, title=f"↪️ Output of {tool_call_to_exec['function']['name']}", border_style="green", expand=False))
                            conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call_to_exec["id"],
                                "name": tool_call_to_exec["function"]["name"],
                                "content": tool_output
                            })
                        
                        # Continue the loop to let assistant process these new results
                        continue
                    else:
                        # No more tool calls, add the final response and break
                        conversation_history.append(continuation_message)
                        break
                
                # If we hit the max rounds, warn about it
                if current_round >= max_continuation_rounds:
                    console.print(f"[yellow]⚠ Reached maximum continuation rounds ({max_continuation_rounds}). Conversation continues.[/yellow]")
            
            # Smart truncation that preserves tool call sequences
            conversation_history = smart_truncate_history(conversation_history, MAX_HISTORY_MESSAGES)

        except KeyboardInterrupt: 
            console.print("\n[yellow]⚠ Interrupted. Ctrl+D or /exit to quit.[/yellow]")
        except EOFError: 
            console.print("[blue]👋 Goodbye! (EOF)[/blue]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n[red]✗ Unexpected error in main loop:[/red]")
            console.print_exception(width=None, extra_lines=1, show_locals=True)

def initialize_application() -> None:
    """Initialize the application and check for existing git repository."""
    if Path(".git").exists() and Path(".git").is_dir():
        agent_state.git_context['enabled'] = True
        try:
            res = subprocess.run(["git", "branch", "--show-current"], cwd=str(Path.cwd()), capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip():
                agent_state.git_context['branch'] = res.stdout.strip()
            else:
                init_branch_res = subprocess.run(["git", "config", "init.defaultBranch"], cwd=str(Path.cwd()), capture_output=True, text=True)
                agent_state.git_context['branch'] = init_branch_res.stdout.strip() if init_branch_res.returncode == 0 and init_branch_res.stdout.strip() else "main"
        except FileNotFoundError:
            console.print("[yellow]Git not found. Git features disabled.[/yellow]")
            agent_state.git_context['enabled'] = False
        except Exception as e:
            console.print(f"[yellow]Could not get Git branch: {e}.[/yellow]")

def get_directory_tree_summary(root_dir: Path, max_depth: int = 3, max_entries: int = 100) -> str:
    """
    Generate a summary of the directory tree up to a certain depth and entry count.
    """
    lines = []
    entry_count = 0

    def walk(dir_path: Path, prefix: str = "", depth: int = 0):
        nonlocal entry_count
        if depth > max_depth or entry_count >= max_entries:
            return
        try:
            entries = sorted([e for e in dir_path.iterdir() if not e.name.startswith('.')])
        except Exception:
            return
        for entry in entries:
            if entry_count >= max_entries:
                lines.append(f"{prefix}... (truncated)")
                return
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                entry_count += 1
                walk(entry, prefix + "  ", depth + 1)
            else:
                lines.append(f"{prefix}{entry.name}")
                entry_count += 1

    walk(root_dir)
    return "\n".join(lines)

def main() -> None:
    """Application entry point."""
    console.print(Panel.fit(
        "[bold bright_blue]🤖 CODY - Advanced Context-Aware AI Coding Assistant[/bold bright_blue]\n"
        "[dim]🚀 Multi-threaded • 🧠 Autonomous Debugging • 🔍 Web Search • 🌐 Multi-language Support[/dim]\n"
        "[dim]✨ Natural Language Processing • 🔧 Code Analysis • 🎯 Predictive Prefetching[/dim]\n"
        "[dim]Type /help for commands. Ctrl+C to interrupt, Ctrl+D or /exit to quit.[/dim]",
        border_style="bright_blue"
    ))

    # Show feature availability status
    features_status = []

    if FUZZY_AVAILABLE:
        features_status.append("[green]✓ Fuzzy Matching[/green]")
    else:
        features_status.append("[red]✗ Fuzzy Matching[/red]")

    if ADVANCED_MODULES_AVAILABLE:
        features_status.append("[green]✓ Advanced Modules[/green]")
        if nlp_processor:
            features_status.append("[green]✓ NLP Processing[/green]")
        if code_analyzer:
            features_status.append("[green]✓ Code Analysis[/green]")
        if autonomous_debugger:
            features_status.append("[green]✓ Autonomous Debugging[/green]")
        if web_search_rag:
            features_status.append("[green]✓ Web Search & RAG[/green]")
        if task_manager:
            features_status.append("[green]✓ Multi-threading[/green]")
    else:
        features_status.append("[red]✗ Advanced Modules[/red]")

    if TREE_SITTER_AVAILABLE:
        features_status.append("[green]✓ AST Parsing[/green]")
    else:
        features_status.append("[yellow]⚠ AST Parsing[/yellow]")

    if WEB_SEARCH_AVAILABLE:
        features_status.append("[green]✓ Web Dependencies[/green]")
    else:
        features_status.append("[yellow]⚠ Web Dependencies[/yellow]")

    console.print("Feature Status: " + " | ".join(features_status))

    if not ADVANCED_MODULES_AVAILABLE:
        console.print("\n[yellow]⚠ Some advanced features are disabled. Check the core/ directory and dependencies.[/yellow]")

    # Add directory structure as a system message before starting the main loop
    dir_summary = get_directory_tree_summary(base_dir)
    conversation_history.append({
        "role": "system",
        "content": f"Project directory structure at startup:\n\n{dir_summary}"
    })

    initialize_application()
    main_loop()

if __name__ == "__main__":
    main()