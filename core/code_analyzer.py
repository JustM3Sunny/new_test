#!/usr/bin/env python3

"""
Code Analysis Module for CODY Agent
Provides AST parsing, static analysis, and code quality assessment
"""

import ast
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

try:
    import pylint.lint
    import pylint.reporters.text
    from io import StringIO
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

logger = logging.getLogger('CODY.CodeAnalyzer')

@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    line_start: int
    line_end: int
    parameters: List[str]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    complexity: int = 0
    is_async: bool = False

@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    line_start: int
    line_end: int
    methods: List[FunctionInfo]
    base_classes: List[str]
    docstring: Optional[str] = None

@dataclass
class CodeIssue:
    """Represents a code issue found during analysis."""
    type: str  # 'error', 'warning', 'info'
    message: str
    line_number: int
    column: Optional[int] = None
    severity: str = 'medium'
    suggestion: Optional[str] = None

class CodeAnalyzer:
    """Advanced code analyzer using AST and static analysis."""
    
    def __init__(self):
        self.supported_languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
        }
        
    def analyze_file(self, file_path: str, analysis_type: str = 'all') -> Dict[str, Any]:
        """
        Analyze a code file and return comprehensive analysis results.
        
        Args:
            file_path: Path to the file to analyze
            analysis_type: Type of analysis ('structure', 'complexity', 'errors', 'all')
            
        Returns:
            Dictionary containing analysis results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        language = self.supported_languages.get(file_ext, 'unknown')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        analysis_result = {
            'file_path': file_path,
            'language': language,
            'lines_of_code': len(content.splitlines()),
            'file_size': len(content),
            'functions': [],
            'classes': [],
            'imports': [],
            'issues': [],
            'complexity_score': 0.0,
            'maintainability_index': 0.0,
        }
        
        if language == 'python':
            analysis_result.update(self._analyze_python_file(content, analysis_type))
        elif language in ['javascript', 'typescript']:
            analysis_result.update(self._analyze_javascript_file(content, analysis_type))
        else:
            analysis_result.update(self._analyze_generic_file(content, analysis_type))
        
        return analysis_result
    
    def _analyze_python_file(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze Python file using AST."""
        result = {
            'functions': [],
            'classes': [],
            'imports': [],
            'issues': [],
            'complexity_score': 0.0,
        }
        
        try:
            tree = ast.parse(content)
            
            if analysis_type in ['structure', 'all']:
                result.update(self._extract_python_structure(tree))
            
            if analysis_type in ['complexity', 'all']:
                result['complexity_score'] = self._calculate_python_complexity(tree)
            
            if analysis_type in ['errors', 'all'] and PYLINT_AVAILABLE:
                result['issues'].extend(self._run_pylint_analysis(content))
            
        except SyntaxError as e:
            result['issues'].append(CodeIssue(
                type='error',
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno or 0,
                column=e.offset,
                severity='high'
            ))
        
        return result
    
    def _extract_python_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract structural information from Python AST."""
        functions = []
        classes = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = FunctionInfo(
                    name=node.name,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    parameters=[arg.arg for arg in node.args.args],
                    docstring=ast.get_docstring(node),
                    is_async=isinstance(node, ast.AsyncFunctionDef)
                )
                functions.append(func_info)
            
            elif isinstance(node, ast.ClassDef):
                class_methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = FunctionInfo(
                            name=item.name,
                            line_start=item.lineno,
                            line_end=item.end_lineno or item.lineno,
                            parameters=[arg.arg for arg in item.args.args],
                            docstring=ast.get_docstring(item)
                        )
                        class_methods.append(method_info)
                
                class_info = ClassInfo(
                    name=node.name,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    methods=class_methods,
                    base_classes=[base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    docstring=ast.get_docstring(node)
                )
                classes.append(class_info)
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                else:
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return {
            'functions': [func.__dict__ for func in functions],
            'classes': [cls.__dict__ for cls in classes],
            'imports': imports
        }
    
    def _calculate_python_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity for Python code."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        
        return complexity / 10.0  # Normalize to 0-1 scale
    
    def _run_pylint_analysis(self, content: str) -> List[CodeIssue]:
        """Run pylint analysis on Python code."""
        issues = []
        
        try:
            # Create a temporary file for pylint
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Capture pylint output
            output = StringIO()
            reporter = pylint.reporters.text.TextReporter(output)
            
            # Run pylint
            pylint.lint.Run([tmp_file_path], reporter=reporter, exit=False)
            
            # Parse pylint output
            pylint_output = output.getvalue()
            for line in pylint_output.split('\n'):
                if ':' in line and any(severity in line for severity in ['ERROR', 'WARNING', 'INFO']):
                    parts = line.split(':')
                    if len(parts) >= 4:
                        try:
                            line_num = int(parts[1])
                            message = ':'.join(parts[3:]).strip()
                            severity = 'high' if 'ERROR' in line else 'medium' if 'WARNING' in line else 'low'
                            
                            issues.append(CodeIssue(
                                type='error' if 'ERROR' in line else 'warning',
                                message=message,
                                line_number=line_num,
                                severity=severity
                            ))
                        except (ValueError, IndexError):
                            continue
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            logger.warning(f"Pylint analysis failed: {e}")
        
        return issues
    
    def _analyze_javascript_file(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript file."""
        result = {
            'functions': [],
            'classes': [],
            'imports': [],
            'issues': [],
            'complexity_score': 0.0,
        }
        
        # Basic regex-based analysis for JavaScript
        # In a real implementation, you'd use a proper JS parser
        
        # Extract functions
        function_pattern = r'(?:function\s+(\w+)|(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))'
        for match in re.finditer(function_pattern, content):
            func_name = match.group(1) or match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            result['functions'].append({
                'name': func_name,
                'line_start': line_num,
                'line_end': line_num,  # Simplified
                'parameters': [],
                'docstring': None
            })
        
        # Extract classes
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            result['classes'].append({
                'name': class_name,
                'line_start': line_num,
                'line_end': line_num,  # Simplified
                'methods': [],
                'base_classes': []
            })
        
        # Extract imports
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)'
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                result['imports'].append(match.group(1))
        
        # Basic complexity calculation
        complexity_indicators = ['if', 'else', 'for', 'while', 'switch', 'case', 'catch']
        complexity = sum(len(re.findall(rf'\b{indicator}\b', content)) for indicator in complexity_indicators)
        result['complexity_score'] = min(1.0, complexity / 20.0)
        
        return result
    
    def _analyze_generic_file(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Generic analysis for unsupported file types."""
        result = {
            'functions': [],
            'classes': [],
            'imports': [],
            'issues': [],
            'complexity_score': 0.0,
        }
        
        # Basic metrics
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in lines if line.strip().startswith(('#', '//', '/*', '*'))]
        
        result['metrics'] = {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'comment_lines': len(comment_lines),
            'blank_lines': len(lines) - len(non_empty_lines)
        }
        
        return result
    
    def search_code_patterns(self, file_path: str, pattern: str, search_type: str = 'regex') -> List[Dict[str, Any]]:
        """
        Search for code patterns in a file.
        
        Args:
            file_path: Path to the file to search
            pattern: Search pattern
            search_type: Type of search ('regex', 'text', 'function', 'class')
            
        Returns:
            List of search results
        """
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = []
        lines = content.splitlines()
        
        if search_type == 'regex':
            for i, line in enumerate(lines):
                matches = re.finditer(pattern, line)
                for match in matches:
                    results.append({
                        'line_number': i + 1,
                        'line_content': line,
                        'match': match.group(),
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    })
        
        elif search_type == 'text':
            for i, line in enumerate(lines):
                if pattern.lower() in line.lower():
                    results.append({
                        'line_number': i + 1,
                        'line_content': line,
                        'match': pattern
                    })
        
        elif search_type in ['function', 'class']:
            # Use the analysis results to find functions/classes
            analysis = self.analyze_file(file_path, 'structure')
            items = analysis.get('functions' if search_type == 'function' else 'classes', [])
            
            for item in items:
                if pattern.lower() in item['name'].lower():
                    results.append({
                        'name': item['name'],
                        'line_start': item['line_start'],
                        'line_end': item.get('line_end', item['line_start']),
                        'type': search_type
                    })
        
        return results
    
    def get_code_suggestions(self, file_path: str) -> List[Dict[str, Any]]:
        """Get code improvement suggestions."""
        analysis = self.analyze_file(file_path, 'all')
        suggestions = []
        
        # Check for long functions
        for func in analysis.get('functions', []):
            if func.get('line_end', 0) - func.get('line_start', 0) > 50:
                suggestions.append({
                    'type': 'refactor',
                    'message': f"Function '{func['name']}' is too long ({func.get('line_end', 0) - func.get('line_start', 0)} lines). Consider breaking it down.",
                    'line_number': func['line_start'],
                    'severity': 'medium'
                })
        
        # Check for missing docstrings
        for func in analysis.get('functions', []):
            if not func.get('docstring'):
                suggestions.append({
                    'type': 'documentation',
                    'message': f"Function '{func['name']}' is missing a docstring.",
                    'line_number': func['line_start'],
                    'severity': 'low'
                })
        
        # Check complexity
        if analysis.get('complexity_score', 0) > 0.7:
            suggestions.append({
                'type': 'complexity',
                'message': "Code complexity is high. Consider refactoring to improve maintainability.",
                'line_number': 1,
                'severity': 'high'
            })
        
        return suggestions
