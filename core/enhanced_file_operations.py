#!/usr/bin/env python3

"""
Enhanced File System Operations for CODY Agent
Comprehensive file handling with regex search, batch operations, and real-time tracking
"""

import os
import re
import time
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Pattern
from dataclasses import dataclass, field
from enum import Enum
import logging
import fnmatch
import hashlib
from collections import defaultdict

logger = logging.getLogger('CODY.EnhancedFileOps')

class FileOperationType(Enum):
    """Types of file operations."""
    CREATE = "create"
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    SEARCH = "search"
    BATCH = "batch"

@dataclass
class FileSearchResult:
    """Result of file search operation."""
    file_path: str
    line_number: int
    line_content: str
    match_text: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    function_name: Optional[str] = None
    class_name: Optional[str] = None

@dataclass
class FileOperationResult:
    """Result of file operation."""
    operation_type: FileOperationType
    file_path: str
    success: bool
    message: str = ""
    content: str = ""
    search_results: List[FileSearchResult] = field(default_factory=list)
    files_affected: List[str] = field(default_factory=list)
    backup_path: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

class EnhancedFileOperations:
    """Enhanced file operations with comprehensive capabilities."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.active_files: Dict[str, str] = {}  # path -> content
        self.file_watchers: Dict[str, float] = {}  # path -> last_modified
        self.backup_dir = self.root_path / ".cody_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # File patterns for different languages
        self.code_patterns = {
            'python': [r'def\s+(\w+)', r'class\s+(\w+)', r'import\s+(\w+)', r'from\s+(\w+)'],
            'javascript': [r'function\s+(\w+)', r'class\s+(\w+)', r'const\s+(\w+)', r'let\s+(\w+)'],
            'java': [r'public\s+class\s+(\w+)', r'public\s+\w+\s+(\w+)\s*\(', r'private\s+\w+\s+(\w+)'],
            'cpp': [r'class\s+(\w+)', r'\w+\s+(\w+)\s*\([^)]*\)\s*{', r'#include\s*[<"]([^>"]+)[>"]']
        }
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    async def create_file(self, file_path: str, content: str = "", backup: bool = True) -> FileOperationResult:
        """Create a new file with optional backup."""
        start_time = time.time()
        
        try:
            path = Path(file_path)
            
            # Create backup if file exists
            backup_path = None
            if backup and path.exists():
                backup_path = await self._create_backup(file_path)
            
            # Create directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Add to active tracking
            with self.lock:
                self.active_files[str(path)] = content
                self.file_watchers[str(path)] = time.time()
            
            execution_time = time.time() - start_time
            
            return FileOperationResult(
                operation_type=FileOperationType.CREATE,
                file_path=str(path),
                success=True,
                message=f"File created successfully",
                content=content,
                backup_path=backup_path,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to create file {file_path}: {e}")
            
            return FileOperationResult(
                operation_type=FileOperationType.CREATE,
                file_path=file_path,
                success=False,
                message=f"Failed to create file: {str(e)}",
                execution_time=execution_time
            )
    
    async def read_file(self, file_path: str) -> FileOperationResult:
        """Read file content with error handling."""
        start_time = time.time()
        
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Update active tracking
            with self.lock:
                self.active_files[str(path)] = content
                self.file_watchers[str(path)] = path.stat().st_mtime
            
            execution_time = time.time() - start_time
            
            return FileOperationResult(
                operation_type=FileOperationType.READ,
                file_path=str(path),
                success=True,
                message=f"File read successfully ({len(content)} characters)",
                content=content,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to read file {file_path}: {e}")
            
            return FileOperationResult(
                operation_type=FileOperationType.READ,
                file_path=file_path,
                success=False,
                message=f"Failed to read file: {str(e)}",
                execution_time=execution_time
            )
    
    async def edit_file(self, file_path: str, new_content: str, backup: bool = True) -> FileOperationResult:
        """Edit existing file with backup."""
        start_time = time.time()
        
        try:
            path = Path(file_path)
            
            # Create backup
            backup_path = None
            if backup and path.exists():
                backup_path = await self._create_backup(file_path)
            
            # Write new content
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Update active tracking
            with self.lock:
                old_content = self.active_files.get(str(path), "")
                self.active_files[str(path)] = new_content
                self.file_watchers[str(path)] = time.time()
            
            execution_time = time.time() - start_time
            
            return FileOperationResult(
                operation_type=FileOperationType.EDIT,
                file_path=str(path),
                success=True,
                message=f"File edited successfully",
                content=new_content,
                backup_path=backup_path,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to edit file {file_path}: {e}")
            
            return FileOperationResult(
                operation_type=FileOperationType.EDIT,
                file_path=file_path,
                success=False,
                message=f"Failed to edit file: {str(e)}",
                execution_time=execution_time
            )
    
    async def search_in_files(self, pattern: str, file_patterns: List[str] = None, 
                            context_lines: int = 3, regex: bool = True) -> FileOperationResult:
        """Advanced search across multiple files with context."""
        start_time = time.time()
        
        try:
            if file_patterns is None:
                file_patterns = ['*.py', '*.js', '*.java', '*.cpp', '*.c', '*.h', '*.txt', '*.md']
            
            search_results = []
            files_searched = []
            
            # Compile regex pattern
            if regex:
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern: {e}")
            
            # Search through files
            for file_pattern in file_patterns:
                for file_path in self.root_path.rglob(file_pattern):
                    if file_path.is_file():
                        files_searched.append(str(file_path))
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()
                            
                            for i, line in enumerate(lines):
                                if regex:
                                    matches = compiled_pattern.finditer(line)
                                    for match in matches:
                                        search_results.append(self._create_search_result(
                                            str(file_path), i + 1, line, match.group(),
                                            lines, i, context_lines
                                        ))
                                else:
                                    if pattern.lower() in line.lower():
                                        search_results.append(self._create_search_result(
                                            str(file_path), i + 1, line, pattern,
                                            lines, i, context_lines
                                        ))
                        
                        except Exception as e:
                            logger.debug(f"Skipped file {file_path}: {e}")
            
            execution_time = time.time() - start_time
            
            return FileOperationResult(
                operation_type=FileOperationType.SEARCH,
                file_path="multiple",
                success=True,
                message=f"Found {len(search_results)} matches in {len(files_searched)} files",
                search_results=search_results,
                files_affected=files_searched,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Search failed: {e}")
            
            return FileOperationResult(
                operation_type=FileOperationType.SEARCH,
                file_path="multiple",
                success=False,
                message=f"Search failed: {str(e)}",
                execution_time=execution_time
            )
    
    async def find_functions(self, language: str = "python", file_patterns: List[str] = None) -> FileOperationResult:
        """Find all functions in project files."""
        start_time = time.time()
        
        try:
            if language not in self.code_patterns:
                raise ValueError(f"Unsupported language: {language}")
            
            if file_patterns is None:
                lang_extensions = {
                    'python': ['*.py'],
                    'javascript': ['*.js', '*.jsx', '*.ts', '*.tsx'],
                    'java': ['*.java'],
                    'cpp': ['*.cpp', '*.c', '*.h', '*.hpp']
                }
                file_patterns = lang_extensions.get(language, ['*.*'])
            
            search_results = []
            files_searched = []
            
            # Get function patterns for the language
            function_patterns = self.code_patterns[language]
            
            for pattern_str in function_patterns:
                pattern = re.compile(pattern_str, re.MULTILINE)
                
                for file_pattern in file_patterns:
                    for file_path in self.root_path.rglob(file_pattern):
                        if file_path.is_file() and str(file_path) not in files_searched:
                            files_searched.append(str(file_path))
                            
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    lines = content.splitlines()
                                
                                for match in pattern.finditer(content):
                                    line_num = content[:match.start()].count('\n') + 1
                                    line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                                    
                                    search_results.append(FileSearchResult(
                                        file_path=str(file_path),
                                        line_number=line_num,
                                        line_content=line_content.strip(),
                                        match_text=match.group(1) if match.groups() else match.group(),
                                        function_name=match.group(1) if match.groups() else None
                                    ))
                            
                            except Exception as e:
                                logger.debug(f"Skipped file {file_path}: {e}")
            
            execution_time = time.time() - start_time
            
            return FileOperationResult(
                operation_type=FileOperationType.SEARCH,
                file_path="multiple",
                success=True,
                message=f"Found {len(search_results)} functions in {len(files_searched)} files",
                search_results=search_results,
                files_affected=files_searched,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function search failed: {e}")
            
            return FileOperationResult(
                operation_type=FileOperationType.SEARCH,
                file_path="multiple",
                success=False,
                message=f"Function search failed: {str(e)}",
                execution_time=execution_time
            )
    
    def _create_search_result(self, file_path: str, line_num: int, line_content: str, 
                            match_text: str, all_lines: List[str], line_index: int, 
                            context_lines: int) -> FileSearchResult:
        """Create a search result with context."""
        context_before = []
        context_after = []
        
        # Get context before
        start_idx = max(0, line_index - context_lines)
        for i in range(start_idx, line_index):
            if i < len(all_lines):
                context_before.append(all_lines[i].rstrip())
        
        # Get context after
        end_idx = min(len(all_lines), line_index + context_lines + 1)
        for i in range(line_index + 1, end_idx):
            if i < len(all_lines):
                context_after.append(all_lines[i].rstrip())
        
        return FileSearchResult(
            file_path=file_path,
            line_number=line_num,
            line_content=line_content.rstrip(),
            match_text=match_text,
            context_before=context_before,
            context_after=context_after
        )
    
    async def _create_backup(self, file_path: str) -> str:
        """Create backup of existing file."""
        path = Path(file_path)
        timestamp = int(time.time())
        backup_name = f"{path.stem}_{timestamp}{path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(path, backup_path)
        return str(backup_path)
    
    def get_active_files(self) -> Dict[str, str]:
        """Get currently active files."""
        with self.lock:
            return self.active_files.copy()
    
    def get_file_changes(self) -> List[str]:
        """Get list of files that have changed."""
        changed_files = []
        
        with self.lock:
            for file_path, last_known_time in self.file_watchers.items():
                try:
                    current_time = Path(file_path).stat().st_mtime
                    if current_time > last_known_time:
                        changed_files.append(file_path)
                        self.file_watchers[file_path] = current_time
                except (FileNotFoundError, OSError):
                    # File was deleted
                    if file_path in self.active_files:
                        del self.active_files[file_path]
                    if file_path in self.file_watchers:
                        del self.file_watchers[file_path]
        
        return changed_files
