#!/usr/bin/env python3

"""
Iterative Execution Engine for CODY Agent
Implements execute → analyze → plan → execute cycle with single-step execution
"""

import asyncio
import time
import subprocess
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

from .enhanced_file_operations import EnhancedFileOperations, FileOperationResult, FileOperationType
from .workflow_engine import ExecutionResult, AnalysisResult, PlanStep, ExecutionPhase, OperationStatus

logger = logging.getLogger('CODY.IterativeEngine')

class CommandType(Enum):
    """Types of commands that can be executed."""
    SYSTEM = "system"
    FILE_OPERATION = "file_operation"
    CODE_GENERATION = "code_generation"
    SEARCH = "search"
    ANALYSIS = "analysis"

@dataclass
class ExecutionContext:
    """Context for execution including user requirements and current state."""
    user_request: str
    current_directory: str
    active_files: Dict[str, str]
    execution_history: List[ExecutionResult]
    analysis_history: List[AnalysisResult]
    goal_achieved: bool = False
    remaining_requirements: List[str] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 10

class IterativeExecutionEngine:
    """Main engine for iterative execution with analysis and planning."""
    
    def __init__(self, file_ops: EnhancedFileOperations):
        self.file_ops = file_ops
        self.execution_context = None
        self.current_phase = ExecutionPhase.EXECUTE
        
        # Natural language command patterns
        self.command_patterns = {
            'time': [r'(?:show|get|what|current)\s+(?:time|date)', r'time\s+(?:now|current)'],
            'list_files': [r'(?:list|show|display)\s+(?:files|directory)', r'ls|dir'],
            'create_file': [r'create\s+(?:file|new\s+file)', r'make\s+(?:file|new\s+file)'],
            'read_file': [r'(?:read|show|display|open)\s+(?:file|content)', r'cat\s+\w+'],
            'search': [r'(?:search|find|grep)\s+(?:for|in)', r'look\s+for'],
            'functions': [r'(?:find|show|list)\s+(?:functions|methods)', r'all\s+functions']
        }
    
    async def execute_iterative_workflow(self, user_request: str) -> Dict[str, Any]:
        """Main iterative workflow execution."""
        # Initialize execution context
        self.execution_context = ExecutionContext(
            user_request=user_request,
            current_directory=os.getcwd(),
            active_files=self.file_ops.get_active_files(),
            execution_history=[],
            analysis_history=[],
            remaining_requirements=[user_request]
        )
        
        workflow_result = {
            'user_request': user_request,
            'iterations': [],
            'final_status': 'in_progress',
            'goal_achieved': False,
            'total_time': 0.0,
            'operations_completed': 0
        }
        
        start_time = time.time()
        
        try:
            while (not self.execution_context.goal_achieved and 
                   self.execution_context.iteration_count < self.execution_context.max_iterations):
                
                iteration_result = await self._execute_single_iteration()
                workflow_result['iterations'].append(iteration_result)
                
                self.execution_context.iteration_count += 1
                
                # Check if goal is achieved
                if iteration_result.get('goal_achieved', False):
                    self.execution_context.goal_achieved = True
                    workflow_result['goal_achieved'] = True
                    workflow_result['final_status'] = 'completed'
                    break
                
                # Safety check for infinite loops
                if self.execution_context.iteration_count >= self.execution_context.max_iterations:
                    workflow_result['final_status'] = 'max_iterations_reached'
                    break
            
            workflow_result['total_time'] = time.time() - start_time
            workflow_result['operations_completed'] = len(self.execution_context.execution_history)
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Iterative workflow failed: {e}")
            workflow_result['final_status'] = 'failed'
            workflow_result['error'] = str(e)
            workflow_result['total_time'] = time.time() - start_time
            return workflow_result
    
    async def _execute_single_iteration(self) -> Dict[str, Any]:
        """Execute a single iteration of the execute → analyze → plan cycle."""
        iteration_start = time.time()
        
        iteration_result = {
            'iteration': self.execution_context.iteration_count + 1,
            'phases': {},
            'goal_achieved': False,
            'next_action': None,
            'execution_time': 0.0
        }
        
        try:
            # Phase 1: EXECUTE
            self.current_phase = ExecutionPhase.EXECUTE
            execution_result = await self._execute_phase()
            iteration_result['phases']['execute'] = execution_result
            
            # Phase 2: ANALYZE
            self.current_phase = ExecutionPhase.ANALYZE
            analysis_result = await self._analyze_phase(execution_result)
            iteration_result['phases']['analyze'] = analysis_result
            
            # Phase 3: PLAN
            self.current_phase = ExecutionPhase.PLAN
            plan_result = await self._plan_phase(analysis_result)
            iteration_result['phases']['plan'] = plan_result
            
            # Phase 4: VALIDATE
            self.current_phase = ExecutionPhase.VALIDATE
            validation_result = await self._validate_phase(plan_result)
            iteration_result['phases']['validate'] = validation_result
            
            # Determine if goal is achieved
            iteration_result['goal_achieved'] = self._check_goal_achievement(analysis_result)
            iteration_result['next_action'] = plan_result.get('next_action')
            
            iteration_result['execution_time'] = time.time() - iteration_start
            
            return iteration_result
            
        except Exception as e:
            logger.error(f"Iteration failed: {e}")
            iteration_result['error'] = str(e)
            iteration_result['execution_time'] = time.time() - iteration_start
            return iteration_result
    
    async def _execute_phase(self) -> Dict[str, Any]:
        """Execute phase - perform the actual operation."""
        phase_start = time.time()
        
        try:
            # Determine what to execute based on current requirements
            if not self.execution_context.remaining_requirements:
                return {'status': 'no_action_needed', 'execution_time': time.time() - phase_start}
            
            current_requirement = self.execution_context.remaining_requirements[0]
            
            # Parse natural language command
            command_info = await self._parse_natural_language_command(current_requirement)
            
            # Execute the command
            execution_result = await self._execute_command(command_info)
            
            # Store execution result
            self.execution_context.execution_history.append(execution_result)
            
            return {
                'status': 'completed',
                'command_info': command_info,
                'execution_result': execution_result.__dict__,
                'execution_time': time.time() - phase_start
            }
            
        except Exception as e:
            logger.error(f"Execute phase failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - phase_start
            }
    
    async def _analyze_phase(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze phase - analyze the execution results."""
        phase_start = time.time()
        
        try:
            if execution_result.get('status') != 'completed':
                return {
                    'status': 'skipped',
                    'reason': 'execution_failed',
                    'execution_time': time.time() - phase_start
                }
            
            exec_result = execution_result.get('execution_result', {})
            
            analysis = AnalysisResult(
                success=exec_result.get('status') == OperationStatus.COMPLETED,
                analysis_time=time.time() - phase_start
            )
            
            # Analyze output and errors
            if exec_result.get('error_message'):
                analysis.errors_found.append(exec_result['error_message'])
                analysis.success = False
            
            # Analyze performance
            exec_time = exec_result.get('execution_time', 0)
            if exec_time > 5.0:  # More than 5 seconds
                analysis.performance_issues.append(f"Slow execution: {exec_time:.2f}s")
            
            # Analyze file system changes
            files_created = exec_result.get('files_created', [])
            files_modified = exec_result.get('files_modified', [])
            
            if files_created:
                analysis.insights.append(f"Created {len(files_created)} files")
                analysis.file_system_changes['created'] = files_created
            
            if files_modified:
                analysis.insights.append(f"Modified {len(files_modified)} files")
                analysis.file_system_changes['modified'] = files_modified
            
            # Determine next actions
            if analysis.success:
                analysis.next_action_recommendations.append("Continue to next requirement")
                analysis.confidence_score = 0.9
            else:
                analysis.next_action_recommendations.append("Retry with error correction")
                analysis.confidence_score = 0.3
            
            # Store analysis result
            self.execution_context.analysis_history.append(analysis)
            
            return {
                'status': 'completed',
                'analysis_result': analysis.__dict__,
                'execution_time': time.time() - phase_start
            }
            
        except Exception as e:
            logger.error(f"Analyze phase failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - phase_start
            }
    
    async def _plan_phase(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Plan phase - determine next actions based on analysis."""
        phase_start = time.time()
        
        try:
            if analysis_result.get('status') != 'completed':
                return {
                    'status': 'skipped',
                    'reason': 'analysis_failed',
                    'execution_time': time.time() - phase_start
                }
            
            analysis = analysis_result.get('analysis_result', {})
            
            # Determine next action
            next_action = None
            
            if analysis.get('success', False):
                # Remove completed requirement
                if self.execution_context.remaining_requirements:
                    completed_req = self.execution_context.remaining_requirements.pop(0)
                    logger.info(f"Completed requirement: {completed_req}")
                
                # Check if there are more requirements
                if self.execution_context.remaining_requirements:
                    next_action = {
                        'type': 'continue',
                        'description': f"Process next requirement: {self.execution_context.remaining_requirements[0]}"
                    }
                else:
                    next_action = {
                        'type': 'complete',
                        'description': "All requirements completed"
                    }
            else:
                # Handle errors
                errors = analysis.get('errors_found', [])
                if errors:
                    next_action = {
                        'type': 'retry',
                        'description': f"Retry with error correction: {errors[0]}"
                    }
                else:
                    next_action = {
                        'type': 'alternative',
                        'description': "Try alternative approach"
                    }
            
            return {
                'status': 'completed',
                'next_action': next_action,
                'remaining_requirements': len(self.execution_context.remaining_requirements),
                'execution_time': time.time() - phase_start
            }
            
        except Exception as e:
            logger.error(f"Plan phase failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - phase_start
            }
    
    async def _validate_phase(self, plan_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate phase - validate the planned actions."""
        phase_start = time.time()
        
        try:
            if plan_result.get('status') != 'completed':
                return {
                    'status': 'skipped',
                    'reason': 'planning_failed',
                    'execution_time': time.time() - phase_start
                }
            
            next_action = plan_result.get('next_action', {})
            
            # Validate the planned action
            validation_result = {
                'valid': True,
                'risk_level': 'low',
                'recommendations': []
            }
            
            action_type = next_action.get('type', '')
            
            if action_type == 'retry':
                validation_result['risk_level'] = 'medium'
                validation_result['recommendations'].append("Monitor for repeated failures")
            elif action_type == 'complete':
                validation_result['recommendations'].append("Verify all requirements are met")
            
            return {
                'status': 'completed',
                'validation_result': validation_result,
                'execution_time': time.time() - phase_start
            }
            
        except Exception as e:
            logger.error(f"Validate phase failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - phase_start
            }
    
    async def _parse_natural_language_command(self, user_input: str) -> Dict[str, Any]:
        """Parse natural language input into executable commands."""
        input_lower = user_input.lower()
        
        # Check for time/date commands
        for pattern in self.command_patterns['time']:
            if re.search(pattern, input_lower):
                return {
                    'type': CommandType.SYSTEM,
                    'command': 'date' if os.name != 'nt' else 'echo %date% %time%',
                    'description': 'Get current date and time'
                }
        
        # Check for file listing
        for pattern in self.command_patterns['list_files']:
            if re.search(pattern, input_lower):
                return {
                    'type': CommandType.SYSTEM,
                    'command': 'ls -la' if os.name != 'nt' else 'dir',
                    'description': 'List files in current directory'
                }
        
        # Check for file creation
        for pattern in self.command_patterns['create_file']:
            if re.search(pattern, input_lower):
                # Extract filename if mentioned
                filename_match = re.search(r'(?:file|named?)\s+["\']?([^\s"\']+)["\']?', input_lower)
                filename = filename_match.group(1) if filename_match else 'new_file.txt'
                
                return {
                    'type': CommandType.FILE_OPERATION,
                    'operation': FileOperationType.CREATE,
                    'file_path': filename,
                    'content': '# New file created by CODY\n',
                    'description': f'Create file: {filename}'
                }
        
        # Check for search operations
        for pattern in self.command_patterns['search']:
            if re.search(pattern, input_lower):
                # Extract search term
                search_match = re.search(r'(?:search|find|grep)\s+(?:for\s+)?["\']?([^"\']+)["\']?', input_lower)
                search_term = search_match.group(1) if search_match else user_input
                
                return {
                    'type': CommandType.SEARCH,
                    'search_term': search_term.strip(),
                    'description': f'Search for: {search_term}'
                }
        
        # Check for function search
        for pattern in self.command_patterns['functions']:
            if re.search(pattern, input_lower):
                return {
                    'type': CommandType.SEARCH,
                    'search_type': 'functions',
                    'language': 'python',  # Default to Python
                    'description': 'Find all functions in project'
                }
        
        # Default: treat as system command
        return {
            'type': CommandType.SYSTEM,
            'command': user_input,
            'description': f'Execute command: {user_input}'
        }
    
    async def _execute_command(self, command_info: Dict[str, Any]) -> ExecutionResult:
        """Execute the parsed command."""
        start_time = time.time()
        operation_id = f"op_{int(time.time() * 1000)}"
        
        try:
            command_type = command_info['type']
            
            if command_type == CommandType.SYSTEM:
                return await self._execute_system_command(operation_id, command_info)
            elif command_type == CommandType.FILE_OPERATION:
                return await self._execute_file_operation(operation_id, command_info)
            elif command_type == CommandType.SEARCH:
                return await self._execute_search_operation(operation_id, command_info)
            else:
                raise ValueError(f"Unsupported command type: {command_type}")
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Command execution failed: {e}")
            
            return ExecutionResult(
                operation_id=operation_id,
                operation_type=str(command_info.get('type', 'unknown')),
                status=OperationStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _execute_system_command(self, operation_id: str, command_info: Dict[str, Any]) -> ExecutionResult:
        """Execute system command."""
        start_time = time.time()
        command = command_info['command']
        
        try:
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.execution_context.current_directory
            )
            
            stdout, stderr = await process.communicate()
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                operation_id=operation_id,
                operation_type="system_command",
                status=OperationStatus.COMPLETED if process.returncode == 0 else OperationStatus.FAILED,
                output=stdout.decode('utf-8', errors='ignore'),
                error_message=stderr.decode('utf-8', errors='ignore') if stderr else "",
                exit_code=process.returncode,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                operation_id=operation_id,
                operation_type="system_command",
                status=OperationStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _execute_file_operation(self, operation_id: str, command_info: Dict[str, Any]) -> ExecutionResult:
        """Execute file operation."""
        start_time = time.time()
        operation = command_info['operation']
        
        try:
            if operation == FileOperationType.CREATE:
                result = await self.file_ops.create_file(
                    command_info['file_path'],
                    command_info.get('content', '')
                )
            else:
                raise ValueError(f"Unsupported file operation: {operation}")
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                operation_id=operation_id,
                operation_type="file_operation",
                status=OperationStatus.COMPLETED if result.success else OperationStatus.FAILED,
                output=result.message,
                error_message="" if result.success else result.message,
                files_created=[result.file_path] if result.success else [],
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                operation_id=operation_id,
                operation_type="file_operation",
                status=OperationStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _execute_search_operation(self, operation_id: str, command_info: Dict[str, Any]) -> ExecutionResult:
        """Execute search operation."""
        start_time = time.time()
        
        try:
            if command_info.get('search_type') == 'functions':
                result = await self.file_ops.find_functions(
                    command_info.get('language', 'python')
                )
            else:
                result = await self.file_ops.search_in_files(
                    command_info['search_term']
                )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                operation_id=operation_id,
                operation_type="search_operation",
                status=OperationStatus.COMPLETED if result.success else OperationStatus.FAILED,
                output=result.message,
                error_message="" if result.success else result.message,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                operation_id=operation_id,
                operation_type="search_operation",
                status=OperationStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _check_goal_achievement(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if the user's goal has been achieved."""
        # Goal is achieved if:
        # 1. No remaining requirements
        # 2. Last operation was successful
        # 3. No critical errors
        
        if not self.execution_context.remaining_requirements:
            return True
        
        analysis = analysis_result.get('analysis_result', {})
        if not analysis.get('success', False):
            return False
        
        # Check for critical errors
        errors = analysis.get('errors_found', [])
        critical_errors = [e for e in errors if 'critical' in e.lower() or 'fatal' in e.lower()]
        
        return len(critical_errors) == 0
