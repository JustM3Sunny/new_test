#!/usr/bin/env python3

"""
Terminal + File System Agent for CODY
Handles terminal commands, file operations, and intelligent task routing
"""

import os
import subprocess
import asyncio
import shlex
import platform
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import queue
import psutil

logger = logging.getLogger('CODY.TerminalFSAgent')

class TaskCategory(Enum):
    """Categories of tasks for intelligent routing."""
    TERMINAL_ONLY = "terminal_only"
    FILE_ONLY = "file_only"
    COMBINED = "combined"
    SETUP_INSTALL = "setup_install"
    GIT_OPERATION = "git_operation"
    BUILD_DEPLOY = "build_deploy"
    TEST_RUN = "test_run"

class CommandType(Enum):
    """Types of terminal commands."""
    SYSTEM = "system"           # ls, cd, mkdir, etc.
    PACKAGE_MANAGER = "package" # npm, pip, apt, etc.
    GIT = "git"                # git commands
    BUILD = "build"            # make, cmake, etc.
    RUN = "run"               # python, node, java, etc.
    TEST = "test"             # pytest, jest, etc.
    DOCKER = "docker"         # docker commands
    CUSTOM = "custom"         # user-defined

@dataclass
class CommandResult:
    """Result of command execution."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    working_directory: str
    timestamp: float = field(default_factory=time.time)
    process_id: Optional[int] = None

@dataclass
class FileOperation:
    """File system operation details."""
    operation: str  # create, read, write, delete, move, copy
    source_path: str
    target_path: Optional[str] = None
    content: Optional[str] = None
    backup_created: bool = False
    timestamp: float = field(default_factory=time.time)

class IntelligentTaskRouter:
    """Routes tasks between terminal, file system, or combined operations."""
    
    def __init__(self):
        self.terminal_patterns = {
            'install': ['install', 'npm install', 'pip install', 'apt install', 'brew install'],
            'run': ['run', 'start', 'execute', 'python', 'node', 'java', 'go run'],
            'build': ['build', 'compile', 'make', 'cmake', 'gradle', 'mvn'],
            'test': ['test', 'pytest', 'jest', 'mocha', 'junit'],
            'git': ['git', 'commit', 'push', 'pull', 'clone', 'branch'],
            'docker': ['docker', 'docker-compose', 'kubectl'],
            'system': ['ls', 'dir', 'cd', 'mkdir', 'rm', 'cp', 'mv', 'find', 'grep']
        }
        
        self.file_patterns = {
            'create': ['create', 'make', 'generate', 'new file', 'touch'],
            'edit': ['edit', 'modify', 'change', 'update', 'write to'],
            'read': ['read', 'show', 'display', 'cat', 'view', 'open'],
            'delete': ['delete', 'remove', 'rm', 'unlink'],
            'move': ['move', 'mv', 'rename'],
            'copy': ['copy', 'cp', 'duplicate']
        }
        
        self.combined_patterns = [
            'setup project', 'initialize', 'scaffold', 'deploy',
            'run tests', 'build and run', 'install and configure'
        ]
    
    def categorize_task(self, user_input: str) -> TaskCategory:
        """Categorize task to determine execution strategy."""
        input_lower = user_input.lower()
        
        # Check for combined operations
        if any(pattern in input_lower for pattern in self.combined_patterns):
            return TaskCategory.COMBINED
        
        # Check for setup/install operations
        if any(word in input_lower for word in ['install', 'setup', 'configure', 'initialize']):
            return TaskCategory.SETUP_INSTALL
        
        # Check for git operations
        if 'git' in input_lower or any(word in input_lower for word in ['commit', 'push', 'pull', 'clone']):
            return TaskCategory.GIT_OPERATION
        
        # Check for build/deploy
        if any(word in input_lower for word in ['build', 'deploy', 'compile', 'package']):
            return TaskCategory.BUILD_DEPLOY
        
        # Check for test operations
        if any(word in input_lower for word in ['test', 'pytest', 'jest', 'junit']):
            return TaskCategory.TEST_RUN
        
        # Check for terminal-only operations
        terminal_score = sum(1 for category, patterns in self.terminal_patterns.items()
                           if any(pattern in input_lower for pattern in patterns))
        
        # Check for file-only operations
        file_score = sum(1 for category, patterns in self.file_patterns.items()
                        if any(pattern in input_lower for pattern in patterns))
        
        if terminal_score > file_score:
            return TaskCategory.TERMINAL_ONLY
        elif file_score > terminal_score:
            return TaskCategory.FILE_ONLY
        else:
            return TaskCategory.COMBINED

class AdvancedTerminalExecutor:
    """Advanced terminal command executor with error detection and recovery."""
    
    def __init__(self):
        self.command_history = []
        self.active_processes = {}
        self.environment_vars = dict(os.environ)
        self.working_directory = os.getcwd()
        self.platform = platform.system().lower()
        
        # Command aliases for cross-platform compatibility
        self.command_aliases = {
            'windows': {
                'ls': 'dir /b',
                'ls -la': 'dir',
                'ls -l': 'dir',
                'cat': 'type',
                'rm': 'del',
                'cp': 'copy',
                'mv': 'move',
                'grep': 'findstr',
                'pwd': 'cd',
                'date': 'date /t',
                'time': 'time /t',
                'ps': 'tasklist',
                'kill': 'taskkill /PID',
                'which': 'where',
                'clear': 'cls',
                'touch': 'echo. >',
                'head': 'more',
                'tail': 'more',
                'find': 'dir /s /b'
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

        # Common commands that work across platforms
        self.universal_commands = {
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
            },
            'system_info': {
                'windows': 'systeminfo | findstr /B /C:"OS Name" /C:"OS Version"',
                'linux': 'uname -a',
                'darwin': 'uname -a'
            }
        }
    
    async def execute_command(self, command: str, timeout: float = 30.0,
                            capture_output: bool = True, shell: bool = True) -> CommandResult:
        """Execute terminal command with advanced error handling and cross-platform support."""
        start_time = time.time()

        # Normalize command for platform
        normalized_command = self._normalize_command(command)
        logger.debug(f"Executing command: {normalized_command} (original: {command})")

        try:
            # Create process with proper encoding and environment
            if self.platform == 'windows':
                # Use cmd.exe for Windows commands
                if not normalized_command.startswith('powershell'):
                    normalized_command = f'cmd /c "{normalized_command}"'

                process = await asyncio.create_subprocess_shell(
                    normalized_command,
                    stdout=asyncio.subprocess.PIPE if capture_output else None,
                    stderr=asyncio.subprocess.PIPE if capture_output else None,
                    cwd=self.working_directory,
                    env=self.environment_vars,
                    shell=True
                )
            else:
                # Use bash for Unix-like systems
                process = await asyncio.create_subprocess_shell(
                    normalized_command,
                    stdout=asyncio.subprocess.PIPE if capture_output else None,
                    stderr=asyncio.subprocess.PIPE if capture_output else None,
                    cwd=self.working_directory,
                    env=self.environment_vars,
                    shell=True
                )

            # Store active process
            self.active_processes[process.pid] = process

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Command timed out after {timeout}s")

            # Clean up
            if process.pid in self.active_processes:
                del self.active_processes[process.pid]

            execution_time = time.time() - start_time

            # Decode output with proper encoding
            stdout_text = ''
            stderr_text = ''

            if stdout:
                try:
                    stdout_text = stdout.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    stdout_text = stdout.decode('cp1252', errors='ignore') if self.platform == 'windows' else stdout.decode('utf-8', errors='ignore')

            if stderr:
                try:
                    stderr_text = stderr.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    stderr_text = stderr.decode('cp1252', errors='ignore') if self.platform == 'windows' else stderr.decode('utf-8', errors='ignore')

            result = CommandResult(
                command=command,
                exit_code=process.returncode,
                stdout=stdout_text.strip(),
                stderr=stderr_text.strip(),
                execution_time=execution_time,
                working_directory=self.working_directory,
                process_id=process.pid
            )

            # Store in history
            self.command_history.append(result)

            # Update working directory if cd command
            if command.strip().startswith('cd '):
                self._update_working_directory(command)

            logger.debug(f"Command completed: exit_code={result.exit_code}, time={execution_time:.2f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Command execution failed: {e}")

            return CommandResult(
                command=command,
                exit_code=-1,
                stdout='',
                stderr=str(e),
                execution_time=execution_time,
                working_directory=self.working_directory
            )
    
    def _normalize_command(self, command: str) -> str:
        """Normalize command for current platform with intelligent mapping."""
        command_lower = command.lower().strip()

        # Handle special universal commands first
        for universal_cmd, platform_cmds in self.universal_commands.items():
            if universal_cmd in command_lower or any(keyword in command_lower for keyword in ['date', 'time', 'pwd', 'current dir']):
                if universal_cmd == 'date_time' and any(word in command_lower for word in ['date', 'time']):
                    return platform_cmds.get(self.platform, command)
                elif universal_cmd == 'current_directory' and any(word in command_lower for word in ['pwd', 'current', 'directory']):
                    return platform_cmds.get(self.platform, command)
                elif universal_cmd == 'list_files' and any(word in command_lower for word in ['ls', 'list', 'files']):
                    return platform_cmds.get(self.platform, command)
                elif universal_cmd == 'system_info' and any(word in command_lower for word in ['system', 'info', 'uname']):
                    return platform_cmds.get(self.platform, command)

        # Handle platform-specific aliases
        if self.platform in self.command_aliases:
            aliases = self.command_aliases[self.platform]

            # Try exact match first
            if command_lower in aliases:
                return aliases[command_lower]

            # Try command with arguments
            try:
                parts = shlex.split(command)
                if parts:
                    base_cmd = parts[0].lower()

                    # Check for exact command match
                    if base_cmd in aliases:
                        # Replace the base command
                        parts[0] = aliases[base_cmd]
                        return ' '.join(parts)

                    # Check for command with common flags
                    cmd_with_flags = ' '.join(parts[:2]) if len(parts) > 1 else base_cmd
                    if cmd_with_flags in aliases:
                        remaining_parts = parts[2:] if len(parts) > 2 else []
                        return aliases[cmd_with_flags] + (' ' + ' '.join(remaining_parts) if remaining_parts else '')
            except ValueError:
                # If shlex fails, fall back to simple replacement
                for alias_cmd, replacement in aliases.items():
                    if command_lower.startswith(alias_cmd):
                        return command.replace(alias_cmd, replacement, 1)

        return command
    
    def _update_working_directory(self, cd_command: str) -> None:
        """Update working directory after cd command."""
        try:
            # Extract path from cd command
            parts = shlex.split(cd_command)
            if len(parts) >= 2:
                new_path = parts[1]
                
                # Handle relative paths
                if not os.path.isabs(new_path):
                    new_path = os.path.join(self.working_directory, new_path)
                
                # Normalize and update
                new_path = os.path.normpath(new_path)
                if os.path.exists(new_path) and os.path.isdir(new_path):
                    self.working_directory = new_path
                    logger.debug(f"Working directory changed to: {new_path}")
        except Exception as e:
            logger.warning(f"Failed to update working directory: {e}")
    
    def detect_errors(self, result: CommandResult) -> List[Dict[str, Any]]:
        """Detect and categorize errors from command execution."""
        errors = []
        
        # Check exit code
        if result.exit_code != 0:
            errors.append({
                'type': 'exit_code',
                'severity': 'high',
                'message': f"Command failed with exit code {result.exit_code}",
                'suggestion': 'Check command syntax and permissions'
            })
        
        # Analyze stderr for common error patterns
        stderr_lower = result.stderr.lower()
        
        error_patterns = {
            'permission_denied': {
                'patterns': ['permission denied', 'access denied', 'not permitted'],
                'suggestion': 'Try running with elevated privileges (sudo/admin)'
            },
            'command_not_found': {
                'patterns': ['command not found', 'is not recognized', 'not found'],
                'suggestion': 'Check if the command is installed and in PATH'
            },
            'file_not_found': {
                'patterns': ['no such file', 'file not found', 'cannot find'],
                'suggestion': 'Verify the file path and ensure the file exists'
            },
            'network_error': {
                'patterns': ['connection refused', 'network unreachable', 'timeout'],
                'suggestion': 'Check network connectivity and firewall settings'
            },
            'syntax_error': {
                'patterns': ['syntax error', 'invalid syntax', 'parse error'],
                'suggestion': 'Review command syntax and fix any typos'
            }
        }
        
        for error_type, info in error_patterns.items():
            if any(pattern in stderr_lower for pattern in info['patterns']):
                errors.append({
                    'type': error_type,
                    'severity': 'medium',
                    'message': f"Detected {error_type.replace('_', ' ')}",
                    'suggestion': info['suggestion']
                })
        
        return errors
    
    def suggest_fixes(self, result: CommandResult, errors: List[Dict[str, Any]]) -> List[str]:
        """Suggest fixes for detected errors."""
        fixes = []
        
        for error in errors:
            if error['type'] == 'permission_denied':
                if self.platform == 'windows':
                    fixes.append(f"Run as administrator: {result.command}")
                else:
                    fixes.append(f"sudo {result.command}")
            
            elif error['type'] == 'command_not_found':
                # Suggest installation commands
                command_name = result.command.split()[0]
                if command_name in ['npm', 'node']:
                    fixes.append("Install Node.js: https://nodejs.org/")
                elif command_name in ['pip', 'python']:
                    fixes.append("Install Python: https://python.org/")
                elif command_name == 'git':
                    fixes.append("Install Git: https://git-scm.com/")
                else:
                    fixes.append(f"Install {command_name} using your package manager")
            
            elif error['type'] == 'file_not_found':
                fixes.append("Check the file path and ensure it exists")
                fixes.append("Use 'ls' or 'dir' to list available files")
        
        return fixes

class SmartFileSystemAgent:
    """Smart file system operations with backup and recovery."""
    
    def __init__(self):
        self.operation_history = []
        self.backup_directory = Path('.cody_backups')
        self.backup_directory.mkdir(exist_ok=True)
    
    async def create_file(self, file_path: str, content: str, backup: bool = True) -> FileOperation:
        """Create a new file with optional backup."""
        path = Path(file_path)
        
        # Create backup if file exists
        backup_created = False
        if backup and path.exists():
            backup_path = self._create_backup(file_path)
            backup_created = True
            logger.info(f"Created backup: {backup_path}")
        
        # Create directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        operation = FileOperation(
            operation='create',
            source_path=file_path,
            content=content,
            backup_created=backup_created
        )
        
        self.operation_history.append(operation)
        logger.info(f"Created file: {file_path}")
        
        return operation
    
    async def read_file(self, file_path: str) -> Tuple[str, FileOperation]:
        """Read file content."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        operation = FileOperation(
            operation='read',
            source_path=file_path,
            content=content
        )
        
        self.operation_history.append(operation)
        return content, operation
    
    async def write_file(self, file_path: str, content: str, backup: bool = True) -> FileOperation:
        """Write content to file with backup."""
        return await self.create_file(file_path, content, backup)
    
    async def delete_file(self, file_path: str, backup: bool = True) -> FileOperation:
        """Delete file with optional backup."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create backup
        backup_created = False
        if backup:
            backup_path = self._create_backup(file_path)
            backup_created = True
            logger.info(f"Created backup before deletion: {backup_path}")
        
        # Delete file
        path.unlink()
        
        operation = FileOperation(
            operation='delete',
            source_path=file_path,
            backup_created=backup_created
        )
        
        self.operation_history.append(operation)
        logger.info(f"Deleted file: {file_path}")
        
        return operation
    
    def _create_backup(self, file_path: str) -> str:
        """Create backup of existing file."""
        path = Path(file_path)
        timestamp = int(time.time())
        backup_name = f"{path.stem}_{timestamp}{path.suffix}"
        backup_path = self.backup_directory / backup_name
        
        # Copy file to backup
        import shutil
        shutil.copy2(path, backup_path)
        
        return str(backup_path)

class TerminalFileSystemAgent:
    """Main agent combining terminal and file system operations."""
    
    def __init__(self):
        self.router = IntelligentTaskRouter()
        self.terminal = AdvancedTerminalExecutor()
        self.file_system = SmartFileSystemAgent()
        
    async def process_task(self, user_input: str) -> Dict[str, Any]:
        """Process user task with intelligent routing."""
        # Categorize task
        category = self.router.categorize_task(user_input)
        
        result = {
            'user_input': user_input,
            'category': category.value,
            'operations': [],
            'success': True,
            'errors': [],
            'suggestions': []
        }
        
        try:
            if category == TaskCategory.TERMINAL_ONLY:
                result.update(await self._handle_terminal_task(user_input))
            elif category == TaskCategory.FILE_ONLY:
                result.update(await self._handle_file_task(user_input))
            else:
                result.update(await self._handle_combined_task(user_input, category))
        
        except Exception as e:
            result['success'] = False
            result['errors'].append(str(e))
            logger.error(f"Task processing failed: {e}")
        
        return result
    
    async def _handle_terminal_task(self, user_input: str) -> Dict[str, Any]:
        """Handle terminal-only tasks."""
        # Extract command from natural language
        command = self._extract_command(user_input)
        
        # Execute command
        cmd_result = await self.terminal.execute_command(command)
        
        # Detect errors
        errors = self.terminal.detect_errors(cmd_result)
        
        # Suggest fixes if needed
        suggestions = []
        if errors:
            suggestions = self.terminal.suggest_fixes(cmd_result, errors)
        
        return {
            'operations': [cmd_result.__dict__],
            'success': cmd_result.exit_code == 0,
            'errors': [e['message'] for e in errors],
            'suggestions': suggestions
        }
    
    async def _handle_file_task(self, user_input: str) -> Dict[str, Any]:
        """Handle file-only tasks."""
        # Parse file operation from natural language
        operation_info = self._parse_file_operation(user_input)
        
        operations = []
        
        if operation_info['operation'] == 'create':
            op = await self.file_system.create_file(
                operation_info['file_path'],
                operation_info.get('content', '')
            )
            operations.append(op.__dict__)
        
        elif operation_info['operation'] == 'read':
            content, op = await self.file_system.read_file(operation_info['file_path'])
            operations.append(op.__dict__)
        
        # Add more file operations as needed
        
        return {
            'operations': operations,
            'success': True,
            'errors': [],
            'suggestions': []
        }
    
    async def _handle_combined_task(self, user_input: str, category: TaskCategory) -> Dict[str, Any]:
        """Handle combined terminal and file tasks."""
        operations = []
        
        if category == TaskCategory.SETUP_INSTALL:
            # Handle setup/installation tasks
            operations.extend(await self._handle_setup_task(user_input))
        
        elif category == TaskCategory.BUILD_DEPLOY:
            # Handle build/deployment tasks
            operations.extend(await self._handle_build_task(user_input))
        
        # Add more combined task handlers
        
        return {
            'operations': operations,
            'success': True,
            'errors': [],
            'suggestions': []
        }
    
    async def _handle_setup_task(self, user_input: str) -> List[Dict[str, Any]]:
        """Handle setup and installation tasks."""
        operations = []
        
        # Example: "setup python project"
        if 'python' in user_input.lower():
            # Create virtual environment
            cmd_result = await self.terminal.execute_command('python -m venv venv')
            operations.append(cmd_result.__dict__)
            
            # Create requirements.txt
            requirements_content = "# Project dependencies\n"
            file_op = await self.file_system.create_file('requirements.txt', requirements_content)
            operations.append(file_op.__dict__)
        
        return operations
    
    async def _handle_build_task(self, user_input: str) -> List[Dict[str, Any]]:
        """Handle build and deployment tasks."""
        operations = []
        
        # Example build task implementation
        if 'build' in user_input.lower():
            cmd_result = await self.terminal.execute_command('npm run build')
            operations.append(cmd_result.__dict__)
        
        return operations
    
    def _extract_command(self, user_input: str) -> str:
        """Extract terminal command from natural language."""
        # Simple extraction - in real implementation, use NLP
        input_lower = user_input.lower()
        
        if 'list files' in input_lower or 'show files' in input_lower:
            return 'ls -la' if self.terminal.platform != 'windows' else 'dir'
        elif 'current directory' in input_lower:
            return 'pwd' if self.terminal.platform != 'windows' else 'cd'
        elif 'install' in input_lower and 'npm' in input_lower:
            return 'npm install'
        elif 'run tests' in input_lower:
            return 'npm test'
        
        # Fallback: assume the input is already a command
        return user_input
    
    def _parse_file_operation(self, user_input: str) -> Dict[str, Any]:
        """Parse file operation from natural language."""
        input_lower = user_input.lower()
        
        # Extract file path
        file_path_match = re.search(r'["\']([^"\']+)["\']', user_input)
        if file_path_match:
            file_path = file_path_match.group(1)
        else:
            # Try to find file-like patterns
            file_match = re.search(r'(\w+\.\w+)', user_input)
            file_path = file_match.group(1) if file_match else 'new_file.txt'
        
        # Determine operation
        if any(word in input_lower for word in ['create', 'make', 'new']):
            operation = 'create'
        elif any(word in input_lower for word in ['read', 'show', 'display']):
            operation = 'read'
        elif any(word in input_lower for word in ['delete', 'remove']):
            operation = 'delete'
        else:
            operation = 'read'  # default
        
        return {
            'operation': operation,
            'file_path': file_path,
            'content': ''  # Could be extracted from user input
        }
