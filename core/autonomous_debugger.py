#!/usr/bin/env python3

"""
Autonomous Debugging Module for CODY Agent
Provides automatic error detection, analysis, and fix suggestions
"""

import re
import ast
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger('CODY.Debugger')

@dataclass
class DebugResult:
    """Result of debugging analysis."""
    error_type: str
    error_message: str
    file_path: str
    line_number: Optional[int]
    suggested_fixes: List[str]
    confidence: float
    auto_fixable: bool
    context: Dict[str, Any]

class AutonomousDebugger:
    """Autonomous debugging system with error detection and fixing capabilities."""
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.fix_templates = self._initialize_fix_templates()
        
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common error patterns and their characteristics."""
        return {
            'python': {
                'SyntaxError': {
                    'patterns': [
                        r"invalid syntax.*line (\d+)",
                        r"unexpected EOF while parsing",
                        r"invalid character.*line (\d+)",
                    ],
                    'common_causes': ['missing parentheses', 'missing colon', 'indentation error'],
                    'auto_fixable': True
                },
                'NameError': {
                    'patterns': [
                        r"name '(\w+)' is not defined",
                        r"global name '(\w+)' is not defined",
                    ],
                    'common_causes': ['undefined variable', 'typo in variable name', 'missing import'],
                    'auto_fixable': True
                },
                'IndentationError': {
                    'patterns': [
                        r"expected an indented block",
                        r"unindent does not match any outer indentation level",
                    ],
                    'common_causes': ['inconsistent indentation', 'missing indentation'],
                    'auto_fixable': True
                },
                'ImportError': {
                    'patterns': [
                        r"No module named '(\w+)'",
                        r"cannot import name '(\w+)'",
                    ],
                    'common_causes': ['missing package', 'wrong import path', 'circular import'],
                    'auto_fixable': False
                },
                'TypeError': {
                    'patterns': [
                        r"'(\w+)' object is not callable",
                        r"unsupported operand type\(s\) for",
                        r"takes (\d+) positional arguments but (\d+) were given",
                    ],
                    'common_causes': ['wrong function call', 'type mismatch', 'argument count mismatch'],
                    'auto_fixable': False
                },
                'AttributeError': {
                    'patterns': [
                        r"'(\w+)' object has no attribute '(\w+)'",
                        r"module '(\w+)' has no attribute '(\w+)'",
                    ],
                    'common_causes': ['typo in attribute name', 'wrong object type', 'missing method'],
                    'auto_fixable': False
                }
            },
            'javascript': {
                'SyntaxError': {
                    'patterns': [
                        r"Unexpected token.*line (\d+)",
                        r"Unexpected end of input",
                    ],
                    'common_causes': ['missing bracket', 'missing semicolon', 'invalid syntax'],
                    'auto_fixable': True
                },
                'ReferenceError': {
                    'patterns': [
                        r"(\w+) is not defined",
                        r"Cannot access '(\w+)' before initialization",
                    ],
                    'common_causes': ['undefined variable', 'hoisting issue', 'scope problem'],
                    'auto_fixable': False
                },
                'TypeError': {
                    'patterns': [
                        r"(\w+) is not a function",
                        r"Cannot read property '(\w+)' of undefined",
                    ],
                    'common_causes': ['wrong function call', 'undefined object', 'type mismatch'],
                    'auto_fixable': False
                }
            }
        }
    
    def _initialize_fix_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize fix templates for common errors."""
        return {
            'python': {
                'SyntaxError': [
                    "Check for missing parentheses, brackets, or quotes",
                    "Verify proper indentation (use 4 spaces)",
                    "Ensure all code blocks end with a colon (:)",
                    "Check for unclosed strings or comments"
                ],
                'NameError': [
                    "Define the variable before using it: {variable} = value",
                    "Check for typos in variable name: {variable}",
                    "Add missing import statement: import {module}",
                    "Ensure variable is in the correct scope"
                ],
                'IndentationError': [
                    "Use consistent indentation (4 spaces recommended)",
                    "Add proper indentation after if/for/while/def/class statements",
                    "Check for mixing tabs and spaces",
                    "Ensure all code blocks are properly indented"
                ],
                'ImportError': [
                    "Install missing package: pip install {package}",
                    "Check import path: from {module} import {name}",
                    "Verify package is installed in current environment",
                    "Check for circular imports"
                ]
            },
            'javascript': {
                'SyntaxError': [
                    "Check for missing brackets, parentheses, or semicolons",
                    "Verify proper string quoting",
                    "Ensure all functions and objects are properly closed",
                    "Check for reserved keyword usage"
                ],
                'ReferenceError': [
                    "Declare variable before use: let {variable} = value",
                    "Check variable scope and hoisting",
                    "Ensure variable is defined in accessible scope",
                    "Use const/let instead of var for block scope"
                ]
            }
        }
    
    def debug_error(self, file_path: str, error_message: str, auto_fix: bool = False) -> DebugResult:
        """
        Debug an error and provide fix suggestions.
        
        Args:
            file_path: Path to the file with the error
            error_message: Error message or description
            auto_fix: Whether to attempt automatic fixing
            
        Returns:
            DebugResult with analysis and suggestions
        """
        # Detect language
        file_ext = Path(file_path).suffix.lower()
        language = self._detect_language(file_ext)
        
        # Analyze error
        error_analysis = self._analyze_error(error_message, language)
        
        # Get file context
        context = self._get_file_context(file_path, error_analysis.get('line_number'))
        
        # Generate fix suggestions
        suggestions = self._generate_fix_suggestions(error_analysis, context, language)
        
        result = DebugResult(
            error_type=error_analysis.get('error_type', 'Unknown'),
            error_message=error_message,
            file_path=file_path,
            line_number=error_analysis.get('line_number'),
            suggested_fixes=suggestions,
            confidence=error_analysis.get('confidence', 0.5),
            auto_fixable=error_analysis.get('auto_fixable', False),
            context=context
        )
        
        # Attempt auto-fix if requested and possible
        if auto_fix and result.auto_fixable and result.confidence > 0.7:
            self._attempt_auto_fix(result)
        
        return result
    
    def _detect_language(self, file_ext: str) -> str:
        """Detect programming language from file extension."""
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
        }
        return language_map.get(file_ext, 'unknown')
    
    def _analyze_error(self, error_message: str, language: str) -> Dict[str, Any]:
        """Analyze error message to extract key information."""
        analysis = {
            'error_type': 'Unknown',
            'line_number': None,
            'confidence': 0.0,
            'auto_fixable': False,
            'variables': [],
            'context_clues': []
        }
        
        if language not in self.error_patterns:
            return analysis
        
        # Try to match error patterns
        for error_type, error_info in self.error_patterns[language].items():
            for pattern in error_info['patterns']:
                match = re.search(pattern, error_message, re.IGNORECASE)
                if match:
                    analysis['error_type'] = error_type
                    analysis['confidence'] = 0.8
                    analysis['auto_fixable'] = error_info['auto_fixable']
                    
                    # Extract line number if present
                    line_match = re.search(r'line (\d+)', error_message)
                    if line_match:
                        analysis['line_number'] = int(line_match.group(1))
                    
                    # Extract variable names if present
                    var_matches = re.findall(r"'(\w+)'", error_message)
                    analysis['variables'] = var_matches
                    
                    break
            
            if analysis['error_type'] != 'Unknown':
                break
        
        return analysis
    
    def _get_file_context(self, file_path: str, line_number: Optional[int]) -> Dict[str, Any]:
        """Get context around the error location."""
        context = {
            'file_exists': False,
            'total_lines': 0,
            'error_line': '',
            'surrounding_lines': [],
            'functions': [],
            'imports': []
        }
        
        try:
            if not Path(file_path).exists():
                return context
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            context['file_exists'] = True
            context['total_lines'] = len(lines)
            
            if line_number and 1 <= line_number <= len(lines):
                context['error_line'] = lines[line_number - 1].strip()
                
                # Get surrounding lines (5 before and after)
                start = max(0, line_number - 6)
                end = min(len(lines), line_number + 5)
                context['surrounding_lines'] = [
                    {'number': i + 1, 'content': lines[i].rstrip()}
                    for i in range(start, end)
                ]
            
            # Extract functions and imports for context
            if file_path.endswith('.py'):
                try:
                    tree = ast.parse(''.join(lines))
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            context['functions'].append(node.name)
                        elif isinstance(node, (ast.Import, ast.ImportFrom)):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    context['imports'].append(alias.name)
                            else:
                                module = node.module or ''
                                context['imports'].append(module)
                except SyntaxError:
                    pass  # File has syntax errors, skip AST parsing
        
        except Exception as e:
            logger.warning(f"Error getting file context: {e}")
        
        return context
    
    def _generate_fix_suggestions(self, error_analysis: Dict[str, Any], context: Dict[str, Any], language: str) -> List[str]:
        """Generate specific fix suggestions based on error analysis."""
        suggestions = []
        error_type = error_analysis.get('error_type', 'Unknown')
        
        # Get base suggestions from templates
        if language in self.fix_templates and error_type in self.fix_templates[language]:
            base_suggestions = self.fix_templates[language][error_type]
            
            # Customize suggestions based on context
            for suggestion in base_suggestions:
                if '{variable}' in suggestion and error_analysis.get('variables'):
                    variable = error_analysis['variables'][0]
                    suggestions.append(suggestion.format(variable=variable))
                elif '{module}' in suggestion and error_analysis.get('variables'):
                    module = error_analysis['variables'][0]
                    suggestions.append(suggestion.format(module=module, package=module))
                else:
                    suggestions.append(suggestion)
        
        # Add context-specific suggestions
        if error_type == 'NameError' and context.get('functions'):
            available_functions = ', '.join(context['functions'][:5])
            suggestions.append(f"Available functions in this file: {available_functions}")
        
        if error_type == 'ImportError' and language == 'python':
            suggestions.append("Check if the package is installed: pip list | grep package_name")
            suggestions.append("Try installing with: pip install package_name")
        
        # Add line-specific suggestions
        if context.get('error_line'):
            error_line = context['error_line']
            
            if error_type == 'SyntaxError':
                if error_line.count('(') != error_line.count(')'):
                    suggestions.append("Check for unmatched parentheses in this line")
                if error_line.count('[') != error_line.count(']'):
                    suggestions.append("Check for unmatched brackets in this line")
                if error_line.count('{') != error_line.count('}'):
                    suggestions.append("Check for unmatched braces in this line")
        
        return suggestions[:10]  # Limit to top 10 suggestions
    
    def _attempt_auto_fix(self, debug_result: DebugResult) -> bool:
        """Attempt to automatically fix simple errors."""
        if not debug_result.auto_fixable or debug_result.confidence < 0.7:
            return False
        
        try:
            with open(debug_result.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            fixed_content = content
            
            # Apply simple fixes based on error type
            if debug_result.error_type == 'IndentationError':
                fixed_content = self._fix_indentation(content)
            elif debug_result.error_type == 'SyntaxError' and 'missing colon' in debug_result.error_message.lower():
                fixed_content = self._fix_missing_colons(content)
            
            # Only write if content actually changed
            if fixed_content != content:
                # Create backup
                backup_path = f"{debug_result.file_path}.backup"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Write fixed content
                with open(debug_result.file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                logger.info(f"Auto-fixed {debug_result.error_type} in {debug_result.file_path}")
                return True
        
        except Exception as e:
            logger.error(f"Auto-fix failed: {e}")
        
        return False
    
    def _fix_indentation(self, content: str) -> str:
        """Fix common indentation issues."""
        lines = content.splitlines()
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Convert tabs to spaces
            line = line.expandtabs(4)
            
            # Fix common indentation patterns
            if i > 0 and lines[i-1].rstrip().endswith(':'):
                # Line after colon should be indented
                if line.strip() and not line.startswith('    '):
                    line = '    ' + line.lstrip()
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_missing_colons(self, content: str) -> str:
        """Fix missing colons in Python control structures."""
        lines = content.splitlines()
        fixed_lines = []
        
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with']
        
        for line in lines:
            stripped = line.strip()
            for keyword in control_keywords:
                if (stripped.startswith(keyword + ' ') or stripped == keyword) and not stripped.endswith(':'):
                    if not any(char in stripped for char in ['#', '"', "'"]):  # Avoid comments and strings
                        line = line.rstrip() + ':'
                        break
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def run_code_and_capture_errors(self, file_path: str) -> Optional[DebugResult]:
        """Run code and capture any runtime errors."""
        if not Path(file_path).exists():
            return None
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.py':
                result = subprocess.run(
                    ['python', file_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    error_output = result.stderr
                    return self.debug_error(file_path, error_output)
            
            elif file_ext == '.js':
                result = subprocess.run(
                    ['node', file_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    error_output = result.stderr
                    return self.debug_error(file_path, error_output)
        
        except subprocess.TimeoutExpired:
            return DebugResult(
                error_type='TimeoutError',
                error_message='Code execution timed out',
                file_path=file_path,
                line_number=None,
                suggested_fixes=['Check for infinite loops', 'Optimize algorithm complexity'],
                confidence=0.8,
                auto_fixable=False,
                context={}
            )
        except Exception as e:
            logger.error(f"Error running code: {e}")
        
        return None
