#!/usr/bin/env python3

"""
Full Codebase Awareness Module for CODY Agent
Indexes all project files, maintains context, and provides intelligent file operations
"""

import os
import time
import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import fnmatch
import mimetypes
from collections import defaultdict, deque
import ast
import re

logger = logging.getLogger('CODY.CodebaseAwareness')

class FileType(Enum):
    """Types of files in the codebase."""
    SOURCE_CODE = "source_code"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    DATA = "data"
    BINARY = "binary"
    UNKNOWN = "unknown"

class FileStatus(Enum):
    """Status of files in the project."""
    ACTIVE = "active"           # Currently being worked on
    TRACKED = "tracked"         # Monitored for changes
    INDEXED = "indexed"         # Scanned and catalogued
    IGNORED = "ignored"         # Explicitly ignored
    DELETED = "deleted"         # Marked for deletion

@dataclass
class FileInfo:
    """Comprehensive information about a file."""
    path: str
    name: str
    extension: str
    size: int
    file_type: FileType
    status: FileStatus = FileStatus.INDEXED
    last_modified: float = 0.0
    last_accessed: float = 0.0
    content_hash: str = ""
    encoding: str = "utf-8"
    line_count: int = 0
    
    # Code-specific information
    language: Optional[str] = None
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    notes: str = ""
    created_at: float = field(default_factory=time.time)

@dataclass
class ProjectStructure:
    """Complete project structure information."""
    root_path: str
    total_files: int = 0
    total_size: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    file_types: Dict[FileType, int] = field(default_factory=dict)
    directory_tree: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    last_scan: float = 0.0

class IntelligentFileIndexer:
    """Intelligent file indexing system."""
    
    def __init__(self):
        self.language_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.sql': 'sql',
            '.sh': 'bash',
            '.ps1': 'powershell',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.md': 'markdown',
            '.txt': 'text'
        }
        
        self.ignore_patterns = [
            '*.pyc', '*.pyo', '*.pyd', '__pycache__',
            'node_modules', '.git', '.svn', '.hg',
            '*.log', '*.tmp', '*.temp',
            '.DS_Store', 'Thumbs.db',
            '*.exe', '*.dll', '*.so', '*.dylib',
            '.env', '.env.local', '.env.production',
            'dist', 'build', 'target', 'bin', 'obj'
        ]
    
    def should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored."""
        file_name = os.path.basename(file_path)
        
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(file_name, pattern) or fnmatch.fnmatch(file_path, pattern):
                return True
        
        return False
    
    def determine_file_type(self, file_path: str) -> FileType:
        """Determine the type of a file."""
        extension = Path(file_path).suffix.lower()
        
        # Source code files
        if extension in self.language_extensions:
            return FileType.SOURCE_CODE
        
        # Configuration files
        config_extensions = ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf']
        config_names = ['Dockerfile', 'Makefile', 'CMakeLists.txt', '.gitignore', '.gitattributes']
        
        if extension in config_extensions or os.path.basename(file_path) in config_names:
            return FileType.CONFIGURATION
        
        # Documentation files
        doc_extensions = ['.md', '.rst', '.txt', '.doc', '.docx', '.pdf']
        if extension in doc_extensions:
            return FileType.DOCUMENTATION
        
        # Data files
        data_extensions = ['.csv', '.json', '.xml', '.sql', '.db', '.sqlite']
        if extension in data_extensions:
            return FileType.DATA
        
        # Binary files
        binary_extensions = ['.exe', '.dll', '.so', '.dylib', '.bin', '.img', '.iso']
        if extension in binary_extensions:
            return FileType.BINARY
        
        return FileType.UNKNOWN
    
    def analyze_source_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Analyze source code file for functions, classes, imports."""
        extension = Path(file_path).suffix.lower()
        language = self.language_extensions.get(extension, 'unknown')
        
        analysis = {
            'language': language,
            'functions': [],
            'classes': [],
            'imports': [],
            'line_count': content.count('\n') + 1
        }
        
        if language == 'python':
            analysis.update(self._analyze_python_file(content))
        elif language in ['javascript', 'typescript']:
            analysis.update(self._analyze_javascript_file(content))
        elif language == 'java':
            analysis.update(self._analyze_java_file(content))
        
        return analysis
    
    def _analyze_python_file(self, content: str) -> Dict[str, Any]:
        """Analyze Python file using AST."""
        try:
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
            
            return {
                'functions': functions,
                'classes': classes,
                'imports': imports
            }
        except SyntaxError:
            return {'functions': [], 'classes': [], 'imports': []}
    
    def _analyze_javascript_file(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript file using regex."""
        functions = []
        classes = []
        imports = []
        
        # Function patterns
        function_patterns = [
            r'function\s+(\w+)',
            r'(\w+)\s*=\s*function',
            r'(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'async\s+function\s+(\w+)',
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, content)
            functions.extend(matches)
        
        # Class patterns
        class_matches = re.findall(r'class\s+(\w+)', content)
        classes.extend(class_matches)
        
        # Import patterns
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)
        
        return {
            'functions': list(set(functions)),
            'classes': list(set(classes)),
            'imports': list(set(imports))
        }
    
    def _analyze_java_file(self, content: str) -> Dict[str, Any]:
        """Analyze Java file using regex."""
        functions = []
        classes = []
        imports = []
        
        # Method patterns
        method_matches = re.findall(r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)', content)
        functions.extend(method_matches)
        
        # Class patterns
        class_matches = re.findall(r'(?:public\s+)?class\s+(\w+)', content)
        classes.extend(class_matches)
        
        # Import patterns
        import_matches = re.findall(r'import\s+([^;]+);', content)
        imports.extend(import_matches)
        
        return {
            'functions': list(set(functions)),
            'classes': list(set(classes)),
            'imports': list(set(imports))
        }

class CodebaseAwareness:
    """Main codebase awareness system."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.indexer = IntelligentFileIndexer()
        
        # File tracking
        self.files: Dict[str, FileInfo] = {}
        self.active_files: Dict[str, str] = {}  # path -> content
        self.file_changes: deque = deque(maxlen=1000)
        
        # Project structure
        self.project_structure = ProjectStructure(root_path=str(self.root_path))
        
        # Monitoring
        self.monitoring = True
        self.last_scan = 0.0
        self.scan_interval = 120.0  # 2 minutes instead of 30 seconds
        
        # Threading
        self.lock = threading.RLock()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Initial scan
        self.full_scan()
    
    def full_scan(self) -> None:
        """Perform a full scan of the codebase."""
        logger.debug(f"Starting full scan of {self.root_path}")  # Changed to debug
        start_time = time.time()

        with self.lock:
            self.files.clear()

            for root, dirs, files in os.walk(self.root_path):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if not self.indexer.should_ignore(os.path.join(root, d))]

                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.root_path)

                    if self.indexer.should_ignore(file_path):
                        continue

                    try:
                        file_info = self._create_file_info(file_path, relative_path)
                        self.files[relative_path] = file_info
                    except Exception as e:
                        logger.warning(f"Failed to index {file_path}: {e}")

            self._update_project_structure()
            self.last_scan = time.time()

        scan_time = time.time() - start_time
        logger.debug(f"Full scan completed in {scan_time:.2f}s. Indexed {len(self.files)} files")  # Changed to debug
    
    def _create_file_info(self, file_path: str, relative_path: str) -> FileInfo:
        """Create comprehensive file information."""
        stat = os.stat(file_path)
        
        file_info = FileInfo(
            path=relative_path,
            name=os.path.basename(file_path),
            extension=Path(file_path).suffix.lower(),
            size=stat.st_size,
            file_type=self.indexer.determine_file_type(file_path),
            last_modified=stat.st_mtime,
            last_accessed=stat.st_atime
        )
        
        # Calculate content hash and analyze if it's a source file
        if file_info.file_type == FileType.SOURCE_CODE and file_info.size < 1024 * 1024:  # < 1MB
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                file_info.content_hash = hashlib.md5(content.encode()).hexdigest()
                
                # Analyze source code
                analysis = self.indexer.analyze_source_file(file_path, content)
                file_info.language = analysis['language']
                file_info.functions = analysis['functions']
                file_info.classes = analysis['classes']
                file_info.imports = analysis['imports']
                file_info.line_count = analysis['line_count']
                
            except Exception as e:
                logger.debug(f"Failed to analyze {file_path}: {e}")
        
        return file_info
    
    def _update_project_structure(self) -> None:
        """Update project structure information."""
        self.project_structure.total_files = len(self.files)
        self.project_structure.total_size = sum(f.size for f in self.files.values())
        
        # Count languages
        languages = defaultdict(int)
        file_types = defaultdict(int)
        
        for file_info in self.files.values():
            if file_info.language:
                languages[file_info.language] += 1
            file_types[file_info.file_type] += 1
        
        self.project_structure.languages = dict(languages)
        self.project_structure.file_types = dict(file_types)
        self.project_structure.last_scan = time.time()
    
    def add_active_file(self, file_path: str, content: str) -> None:
        """Add a file to active tracking."""
        with self.lock:
            self.active_files[file_path] = content
            
            # Update or create file info
            if file_path in self.files:
                self.files[file_path].status = FileStatus.ACTIVE
                self.files[file_path].last_accessed = time.time()
            else:
                # Create new file info
                full_path = os.path.join(self.root_path, file_path)
                if os.path.exists(full_path):
                    file_info = self._create_file_info(full_path, file_path)
                    file_info.status = FileStatus.ACTIVE
                    self.files[file_path] = file_info
        
        logger.debug(f"Added active file: {file_path}")
    
    def update_file_content(self, file_path: str, new_content: str) -> Dict[str, Any]:
        """Update file content and track changes."""
        with self.lock:
            old_content = self.active_files.get(file_path, "")
            self.active_files[file_path] = new_content
            
            # Calculate change metrics
            change_info = {
                'file_path': file_path,
                'timestamp': time.time(),
                'old_size': len(old_content),
                'new_size': len(new_content),
                'lines_added': new_content.count('\n') - old_content.count('\n'),
                'content_hash': hashlib.md5(new_content.encode()).hexdigest()
            }
            
            # Update file info
            if file_path in self.files:
                file_info = self.files[file_path]
                file_info.content_hash = change_info['content_hash']
                file_info.last_modified = time.time()
                file_info.size = len(new_content)
                file_info.line_count = new_content.count('\n') + 1
                
                # Re-analyze if it's source code
                if file_info.file_type == FileType.SOURCE_CODE:
                    analysis = self.indexer.analyze_source_file(file_path, new_content)
                    file_info.functions = analysis['functions']
                    file_info.classes = analysis['classes']
                    file_info.imports = analysis['imports']
            
            self.file_changes.append(change_info)
            return change_info
    
    def get_active_files(self) -> List[str]:
        """Get list of currently active files."""
        with self.lock:
            return list(self.active_files.keys())
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of an active file."""
        with self.lock:
            return self.active_files.get(file_path)

    def get_file_changes(self) -> List[Dict[str, Any]]:
        """Get recent file changes."""
        with self.lock:
            return list(self.file_changes)
    
    def search_files(self, query: str, file_type: Optional[FileType] = None, 
                    language: Optional[str] = None) -> List[FileInfo]:
        """Search files by various criteria."""
        results = []
        
        with self.lock:
            for file_info in self.files.values():
                # Filter by file type
                if file_type and file_info.file_type != file_type:
                    continue
                
                # Filter by language
                if language and file_info.language != language:
                    continue
                
                # Search in file name, functions, classes
                search_targets = [
                    file_info.name.lower(),
                    file_info.path.lower(),
                    *[f.lower() for f in file_info.functions],
                    *[c.lower() for c in file_info.classes]
                ]
                
                if any(query.lower() in target for target in search_targets):
                    results.append(file_info)
        
        return results
    
    def get_project_summary(self) -> Dict[str, Any]:
        """Get comprehensive project summary."""
        with self.lock:
            active_count = len(self.active_files)
            recent_changes = len([c for c in self.file_changes if time.time() - c['timestamp'] < 3600])
            
            return {
                'project_structure': self.project_structure.__dict__,
                'active_files': active_count,
                'recent_changes': recent_changes,
                'total_functions': sum(len(f.functions) for f in self.files.values()),
                'total_classes': sum(len(f.classes) for f in self.files.values()),
                'languages_used': list(self.project_structure.languages.keys()),
                'largest_files': sorted(
                    [(f.path, f.size) for f in self.files.values()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                current_time = time.time()
                
                # Periodic full scan
                if current_time - self.last_scan > self.scan_interval:
                    self.full_scan()
                
                # Check for file changes
                self._check_file_changes()

                time.sleep(30)  # Check every 30 seconds instead of 5
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    def _check_file_changes(self) -> None:
        """Check for changes in tracked files."""
        with self.lock:
            for file_path, file_info in self.files.items():
                if file_info.status == FileStatus.ACTIVE:
                    full_path = os.path.join(self.root_path, file_path)
                    
                    if os.path.exists(full_path):
                        stat = os.stat(full_path)
                        if stat.st_mtime > file_info.last_modified:
                            # File was modified externally
                            try:
                                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    new_content = f.read()
                                
                                self.update_file_content(file_path, new_content)
                                logger.debug(f"Detected external change: {file_path}")
                                
                            except Exception as e:
                                logger.warning(f"Failed to read changed file {file_path}: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the codebase awareness system."""
        self.monitoring = False
        logger.info("Codebase awareness system shutdown")
