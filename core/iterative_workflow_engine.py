#!/usr/bin/env python3

"""
Iterative Workflow Engine for CODY Agent
Implements the complete execute → analyze → plan → execute cycle following workflow.md
"""

import asyncio
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger('CODY.IterativeWorkflow')

class WorkflowStage(Enum):
    """Stages in the iterative workflow following workflow.md diagram."""
    USER_INPUT = "user_input"
    NLP_UNDERSTANDING = "nlp_understanding"
    CONTEXT_ENGINE = "context_engine"
    INTENT_ANALYSIS = "intent_analysis"
    TASK_ROUTING = "task_routing"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    VALIDATION = "validation"
    COMPLETION = "completion"

class TaskType(Enum):
    """Types of tasks that can be executed."""
    TERMINAL_COMMAND = "terminal_command"
    FILE_OPERATION = "file_operation"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    DEBUGGING = "debugging"
    WEB_SEARCH = "web_search"
    SYSTEM_INFO = "system_info"
    MIXED_OPERATION = "mixed_operation"

@dataclass
class WorkflowStep:
    """Represents a single step in the iterative workflow."""
    step_id: str
    stage: WorkflowStage
    task_type: TaskType
    description: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = False
    errors: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class WorkflowContext:
    """Complete context for the iterative workflow."""
    user_request: str
    original_intent: str
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    active_files: List[str] = field(default_factory=list)
    terminal_history: List[Dict[str, Any]] = field(default_factory=list)
    current_directory: str = "."
    project_state: Dict[str, Any] = field(default_factory=dict)
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    accumulated_knowledge: Dict[str, Any] = field(default_factory=dict)
    goal_progress: float = 0.0
    remaining_tasks: List[str] = field(default_factory=list)

class NaturalLanguageProcessor:
    """Enhanced NLP processor for understanding user intent."""
    
    def __init__(self):
        self.intent_patterns = {
            TaskType.TERMINAL_COMMAND: [
                r'\b(run|execute|command|terminal|shell|bash|cmd)\b',
                r'\b(show|display|get|check)\s+(date|time|current|directory|files|status)\b',
                r'\b(list|ls|dir)\b',
                r'\b(pwd|cd|mkdir|rm|cp|mv)\b'
            ],
            TaskType.FILE_OPERATION: [
                r'\b(create|make|generate|new)\s+(file|folder|directory)\b',
                r'\b(read|open|view|show|display)\s+(file|content)\b',
                r'\b(edit|modify|change|update|write)\s+(file|code)\b',
                r'\b(delete|remove|rm)\s+(file|folder)\b'
            ],
            TaskType.CODE_GENERATION: [
                r'\b(write|create|generate|build)\s+(code|function|class|script|program)\b',
                r'\b(implement|develop|code)\b',
                r'\b(python|javascript|java|cpp|go|rust)\s+(code|script|function)\b'
            ],
            TaskType.CODE_ANALYSIS: [
                r'\b(analyze|review|check|examine|inspect)\s+(code|file|function)\b',
                r'\b(find|search|look for)\s+(function|class|variable|pattern)\b',
                r'\b(lint|format|style|quality)\b'
            ],
            TaskType.DEBUGGING: [
                r'\b(debug|fix|error|bug|issue|problem)\b',
                r'\b(troubleshoot|diagnose|resolve)\b',
                r'\b(exception|traceback|stack trace)\b'
            ],
            TaskType.WEB_SEARCH: [
                r'\b(search|find|look up|google|documentation|docs)\b',
                r'\b(how to|tutorial|example|guide)\b',
                r'\b(stackoverflow|github|official docs)\b'
            ],
            TaskType.SYSTEM_INFO: [
                r'\b(system|os|platform|environment|version)\s+(info|information|details)\b',
                r'\b(check|show|display)\s+(system|environment|config)\b'
            ]
        }
    
    def analyze_intent(self, user_input: str) -> Tuple[TaskType, float, Dict[str, Any]]:
        """Analyze user input to determine intent and extract entities."""
        user_input_lower = user_input.lower()

        # Score each task type
        scores = {}
        entities = {}

        for task_type, patterns in self.intent_patterns.items():
            score = 0
            matches = []

            for pattern in patterns:
                pattern_matches = re.findall(pattern, user_input_lower)
                if pattern_matches:
                    score += len(pattern_matches)
                    matches.extend(pattern_matches)

            if score > 0:
                scores[task_type] = score
                entities[task_type.value] = matches

        # Determine primary intent
        if scores:
            primary_task = max(scores, key=scores.get)
            confidence = min(scores[primary_task] / 3.0, 1.0)  # Normalize to 0-1
        else:
            primary_task = TaskType.MIXED_OPERATION
            confidence = 0.5

        # Extract specific entities
        extracted_entities = self._extract_entities(user_input)
        entities.update(extracted_entities)

        return primary_task, confidence, entities

    def process_natural_language(self, user_input: str):
        """Compatibility method for existing NLP processor interface."""
        task_type, confidence, entities = self.analyze_intent(user_input)

        # Create a simple result object that matches the expected interface
        class NLPResult:
            def __init__(self, intent, confidence, entities):
                self.intent = intent
                self.confidence = confidence
                self.entities = entities

        return NLPResult(task_type, confidence, entities)
    
    def _extract_entities(self, user_input: str) -> Dict[str, Any]:
        """Extract specific entities from user input."""
        entities = {}
        
        # Extract file paths and names
        file_patterns = [
            r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']',  # Quoted file names
            r'\b(\w+\.[a-zA-Z0-9]+)\b',  # Simple file names
            r'([./]\S+)',  # Path-like strings
        ]
        
        files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, user_input)
            files.extend(matches)
        
        if files:
            entities['files'] = list(set(files))
        
        # Extract programming languages
        languages = re.findall(r'\b(python|javascript|typescript|java|cpp|c\+\+|go|rust|php|ruby|swift|kotlin)\b', user_input.lower())
        if languages:
            entities['languages'] = list(set(languages))
        
        # Extract commands
        commands = re.findall(r'\b(ls|dir|pwd|cd|mkdir|rm|cp|mv|cat|grep|find|ps|kill|date|time)\b', user_input.lower())
        if commands:
            entities['commands'] = list(set(commands))
        
        return entities

class IterativeAnalyzer:
    """Analyzes execution results and determines next steps."""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_execution_result(self, step: WorkflowStep, context: WorkflowContext) -> Dict[str, Any]:
        """Perform comprehensive analysis of execution results."""
        analysis = {
            'step_id': step.step_id,
            'success': step.success,
            'execution_time': step.execution_time,
            'insights': [],
            'issues': [],
            'recommendations': [],
            'goal_progress': 0.0,
            'next_actions': []
        }
        
        # Analyze based on task type
        if step.task_type == TaskType.TERMINAL_COMMAND:
            analysis.update(self._analyze_terminal_result(step))
        elif step.task_type == TaskType.FILE_OPERATION:
            analysis.update(self._analyze_file_result(step))
        elif step.task_type == TaskType.CODE_GENERATION:
            analysis.update(self._analyze_code_result(step))
        
        # Calculate goal progress
        analysis['goal_progress'] = self._calculate_goal_progress(step, context)
        
        # Determine next actions
        analysis['next_actions'] = self._determine_next_actions(step, context, analysis)
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _analyze_terminal_result(self, step: WorkflowStep) -> Dict[str, Any]:
        """Analyze terminal command execution results."""
        result = step.output_data
        analysis = {'insights': [], 'issues': [], 'recommendations': []}
        
        if result.get('exit_code') == 0:
            analysis['insights'].append("Command executed successfully")
            
            # Analyze output content
            stdout = result.get('stdout', '')
            if stdout:
                if 'date' in step.description.lower() or 'time' in step.description.lower():
                    analysis['insights'].append(f"Current date/time retrieved: {stdout.strip()}")
                elif 'directory' in step.description.lower() or 'pwd' in step.description.lower():
                    analysis['insights'].append(f"Current directory: {stdout.strip()}")
                elif 'list' in step.description.lower() or 'ls' in step.description.lower():
                    files = stdout.strip().split('\n')
                    analysis['insights'].append(f"Found {len(files)} items in directory")
        else:
            analysis['issues'].append(f"Command failed with exit code {result.get('exit_code')}")
            stderr = result.get('stderr', '')
            if stderr:
                analysis['issues'].append(f"Error: {stderr}")
                analysis['recommendations'].append("Check command syntax and permissions")
        
        return analysis
    
    def _analyze_file_result(self, step: WorkflowStep) -> Dict[str, Any]:
        """Analyze file operation results."""
        analysis = {'insights': [], 'issues': [], 'recommendations': []}
        
        if step.success:
            operation = step.output_data.get('operation', '')
            file_path = step.output_data.get('source_path', '')
            
            if operation == 'create':
                analysis['insights'].append(f"Successfully created file: {file_path}")
                analysis['recommendations'].append("Consider adding the file to version control")
            elif operation == 'read':
                content_length = len(step.output_data.get('content', ''))
                analysis['insights'].append(f"Read file with {content_length} characters")
            elif operation == 'edit':
                analysis['insights'].append(f"Successfully modified file: {file_path}")
                analysis['recommendations'].append("Review changes and test functionality")
        else:
            analysis['issues'].extend(step.errors)
            analysis['recommendations'].append("Check file permissions and path validity")
        
        return analysis
    
    def _analyze_code_result(self, step: WorkflowStep) -> Dict[str, Any]:
        """Analyze code generation results."""
        analysis = {'insights': [], 'issues': [], 'recommendations': []}
        
        if step.success:
            code = step.output_data.get('code', '')
            if code:
                lines = code.count('\n') + 1
                analysis['insights'].append(f"Generated {lines} lines of code")
                
                # Basic code quality checks
                if 'def ' in code or 'function ' in code:
                    analysis['insights'].append("Code includes function definitions")
                if 'class ' in code:
                    analysis['insights'].append("Code includes class definitions")
                if 'import ' in code or 'require(' in code:
                    analysis['insights'].append("Code includes dependency imports")
                
                analysis['recommendations'].extend([
                    "Review generated code for correctness",
                    "Add unit tests for new functionality",
                    "Consider code documentation and comments"
                ])
        else:
            analysis['issues'].extend(step.errors)
            analysis['recommendations'].append("Refine requirements and try again")
        
        return analysis
    
    def _calculate_goal_progress(self, step: WorkflowStep, context: WorkflowContext) -> float:
        """Calculate progress toward the user's goal."""
        # Simple heuristic based on successful steps
        total_steps = len(context.execution_results) + 1
        successful_steps = sum(1 for result in context.execution_results if result.get('success', False))
        
        if step.success:
            successful_steps += 1
        
        return min(successful_steps / max(total_steps, 1), 1.0)
    
    def _determine_next_actions(self, step: WorkflowStep, context: WorkflowContext, analysis: Dict[str, Any]) -> List[str]:
        """Determine optimal next actions based on analysis."""
        next_actions = []
        
        if step.success:
            # Suggest follow-up actions based on task type
            if step.task_type == TaskType.TERMINAL_COMMAND:
                if 'date' in step.description.lower():
                    next_actions.append("Consider checking system timezone if needed")
                elif 'directory' in step.description.lower():
                    next_actions.append("Explore directory contents if needed")
            elif step.task_type == TaskType.FILE_OPERATION:
                if step.output_data.get('operation') == 'create':
                    next_actions.append("Add content to the newly created file")
                    next_actions.append("Test the file functionality")
            elif step.task_type == TaskType.CODE_GENERATION:
                next_actions.extend([
                    "Test the generated code",
                    "Add error handling if needed",
                    "Consider performance optimizations"
                ])
        else:
            # Suggest recovery actions
            next_actions.extend([
                "Analyze error messages for root cause",
                "Try alternative approach",
                "Check system requirements and permissions"
            ])
        
        # Check if goal is achieved
        if analysis['goal_progress'] >= 0.8:
            next_actions.append("Goal appears to be largely achieved")
        elif analysis['goal_progress'] < 0.3:
            next_actions.append("Consider breaking down the task into smaller steps")
        
        return next_actions

class IntelligentPlanner:
    """Plans next steps based on analysis results and user goals."""
    
    def __init__(self):
        self.planning_history = []
    
    def create_execution_plan(self, user_request: str, context: WorkflowContext, 
                            analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an intelligent execution plan based on analysis."""
        plan = {
            'plan_id': f"plan_{int(time.time())}",
            'user_request': user_request,
            'strategy': 'iterative',
            'steps': [],
            'estimated_completion': 0.0,
            'risk_factors': [],
            'success_criteria': []
        }
        
        # Analyze current state
        current_progress = context.goal_progress
        recent_failures = sum(1 for result in context.execution_results[-3:] if not result.get('success', True))
        
        # Determine strategy
        if recent_failures > 1:
            plan['strategy'] = 'conservative'
            plan['risk_factors'].append("Recent execution failures detected")
        elif current_progress > 0.7:
            plan['strategy'] = 'completion'
        else:
            plan['strategy'] = 'progressive'
        
        # Generate next steps
        next_steps = self._generate_next_steps(user_request, context, analysis_results)
        plan['steps'] = next_steps
        
        # Estimate completion
        plan['estimated_completion'] = self._estimate_completion_time(next_steps)
        
        # Define success criteria
        plan['success_criteria'] = self._define_success_criteria(user_request, context)
        
        self.planning_history.append(plan)
        return plan
    
    def _generate_next_steps(self, user_request: str, context: WorkflowContext, 
                           analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate specific next steps based on current state."""
        steps = []
        
        # If no previous execution, start with intent analysis
        if not context.execution_results:
            steps.append({
                'step_type': 'intent_analysis',
                'description': 'Analyze user intent and extract requirements',
                'priority': 'high',
                'estimated_time': 5
            })
        
        # Check if we need more information
        if context.goal_progress < 0.3:
            steps.append({
                'step_type': 'information_gathering',
                'description': 'Gather additional context and requirements',
                'priority': 'medium',
                'estimated_time': 10
            })
        
        # Add task-specific steps based on latest analysis
        if analysis_results:
            latest_analysis = analysis_results[-1]
            for action in latest_analysis.get('next_actions', []):
                steps.append({
                    'step_type': 'execution',
                    'description': action,
                    'priority': 'medium',
                    'estimated_time': 15
                })
        
        # Add validation step if we're near completion
        if context.goal_progress > 0.6:
            steps.append({
                'step_type': 'validation',
                'description': 'Validate results against user requirements',
                'priority': 'high',
                'estimated_time': 10
            })
        
        return steps
    
    def _estimate_completion_time(self, steps: List[Dict[str, Any]]) -> float:
        """Estimate total completion time for the plan."""
        return sum(step.get('estimated_time', 15) for step in steps)
    
    def _define_success_criteria(self, user_request: str, context: WorkflowContext) -> List[str]:
        """Define success criteria for the plan."""
        criteria = ["User request fulfilled successfully"]
        
        # Add specific criteria based on request type
        if any(word in user_request.lower() for word in ['create', 'generate', 'make']):
            criteria.append("Required files/code created and functional")
        
        if any(word in user_request.lower() for word in ['fix', 'debug', 'error']):
            criteria.append("Errors resolved and system working correctly")
        
        if any(word in user_request.lower() for word in ['show', 'display', 'get']):
            criteria.append("Requested information displayed to user")
        
        criteria.append("No critical errors in execution")
        criteria.append("User satisfaction confirmed")
        
        return criteria

class IterativeWorkflowEngine:
    """Main iterative workflow engine implementing execute → analyze → plan → execute cycle."""
    
    def __init__(self, terminal_executor, file_manager, nlp_processor=None):
        self.terminal_executor = terminal_executor
        self.file_manager = file_manager
        self.nlp_processor = nlp_processor or NaturalLanguageProcessor()
        self.analyzer = IterativeAnalyzer()
        self.planner = IntelligentPlanner()
        
        self.workflow_history = []
        self.active_workflows = {}
    
    async def process_user_request(self, user_request: str, context: WorkflowContext) -> Dict[str, Any]:
        """Process user request through the complete iterative workflow."""
        workflow_id = f"workflow_{int(time.time())}"

        try:
            # Stage 1: Natural Language Understanding
            # Handle both old and new NLP processor interfaces with better error handling
            try:
                if hasattr(self.nlp_processor, 'analyze_intent'):
                    task_type, confidence, entities = self.nlp_processor.analyze_intent(user_request)
                elif hasattr(self.nlp_processor, 'process_natural_language'):
                    # Use the old interface
                    nlp_result = self.nlp_processor.process_natural_language(user_request)
                    # Map old interface to new
                    task_type = self._map_intent_to_task_type(nlp_result.intent)
                    confidence = nlp_result.confidence
                    entities = getattr(nlp_result, 'entities', {})
                else:
                    # Fallback to simple analysis
                    task_type, confidence, entities = self._simple_intent_analysis(user_request)
            except Exception as nlp_error:
                print(f"[yellow]⚠ NLP processing error: {nlp_error}[/yellow]")
                # Fallback to simple analysis
                task_type, confidence, entities = self._simple_intent_analysis(user_request)

            # Stage 2: Context Engine - Update context with new information
            context.user_request = user_request
            context.original_intent = task_type.value if hasattr(task_type, 'value') else str(task_type)

            # Stage 3: Intent Analysis and Task Routing
            execution_plan = self.planner.create_execution_plan(user_request, context, [])

            # Stage 4: Iterative Execution Loop
            workflow_result = await self._execute_iterative_loop(
                workflow_id, task_type, entities, context, execution_plan
            )

            # Store workflow
            self.workflow_history.append({
                'workflow_id': workflow_id,
                'user_request': user_request,
                'result': workflow_result,
                'timestamp': time.time()
            })

            return workflow_result

        except Exception as e:
            logger.error(f"Workflow processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'workflow_id': workflow_id
            }

    def _map_intent_to_task_type(self, intent):
        """Map old intent enum to new task type."""
        intent_str = str(intent).lower()

        if 'terminal' in intent_str or 'command' in intent_str:
            return TaskType.TERMINAL_COMMAND
        elif 'file' in intent_str:
            return TaskType.FILE_OPERATION
        elif 'code' in intent_str:
            return TaskType.CODE_GENERATION
        elif 'debug' in intent_str:
            return TaskType.DEBUGGING
        elif 'search' in intent_str:
            return TaskType.WEB_SEARCH
        else:
            return TaskType.MIXED_OPERATION

    def _simple_intent_analysis(self, user_request: str) -> Tuple[TaskType, float, Dict[str, Any]]:
        """Simple fallback intent analysis."""
        user_lower = user_request.lower()

        # Simple keyword matching
        if any(word in user_lower for word in ['date', 'time', 'terminal', 'command', 'show', 'display']):
            return TaskType.TERMINAL_COMMAND, 0.8, {'commands': ['date']}
        elif any(word in user_lower for word in ['file', 'create', 'read', 'write']):
            return TaskType.FILE_OPERATION, 0.7, {}
        elif any(word in user_lower for word in ['code', 'function', 'class']):
            return TaskType.CODE_GENERATION, 0.7, {}
        else:
            return TaskType.MIXED_OPERATION, 0.5, {}
    
    async def _execute_iterative_loop(self, workflow_id: str, task_type: TaskType, 
                                    entities: Dict[str, Any], context: WorkflowContext,
                                    execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main iterative loop: execute → analyze → plan → execute."""
        max_iterations = 5
        current_iteration = 0
        analysis_results = []
        
        while current_iteration < max_iterations and context.goal_progress < 0.9:
            current_iteration += 1
            
            # EXECUTE: Perform the current task
            step = await self._execute_single_step(task_type, entities, context)
            context.execution_results.append(step.__dict__)
            
            # ANALYZE: Analyze the execution results
            analysis = self.analyzer.analyze_execution_result(step, context)
            analysis_results.append(analysis)
            
            # Update context with new insights
            context.goal_progress = analysis['goal_progress']
            context.accumulated_knowledge.update({
                f'step_{step.step_id}': analysis['insights']
            })
            
            # PLAN: Determine next actions
            if context.goal_progress < 0.9 and current_iteration < max_iterations:
                updated_plan = self.planner.create_execution_plan(
                    context.user_request, context, analysis_results
                )
                
                # Determine next task type based on plan
                if updated_plan['steps']:
                    next_step = updated_plan['steps'][0]
                    # Update task type for next iteration if needed
                    # This is where we could switch between different task types
            
            # Check if we should continue
            if step.success and analysis['goal_progress'] >= 0.9:
                break
            elif not step.success and current_iteration >= 2:
                # If we've failed multiple times, try a different approach
                break
        
        return {
            'success': context.goal_progress >= 0.7,
            'workflow_id': workflow_id,
            'iterations': current_iteration,
            'final_progress': context.goal_progress,
            'execution_results': context.execution_results,
            'analysis_results': analysis_results,
            'accumulated_knowledge': context.accumulated_knowledge
        }
    
    async def _execute_single_step(self, task_type: TaskType, entities: Dict[str, Any], 
                                 context: WorkflowContext) -> WorkflowStep:
        """Execute a single step in the workflow."""
        step_id = f"step_{len(context.execution_results) + 1}"
        step = WorkflowStep(
            step_id=step_id,
            stage=WorkflowStage.EXECUTION,
            task_type=task_type,
            description=f"Execute {task_type.value}"
        )
        
        start_time = time.time()
        
        try:
            if task_type == TaskType.TERMINAL_COMMAND:
                result = await self._execute_terminal_task(entities, context)
            elif task_type == TaskType.FILE_OPERATION:
                result = await self._execute_file_task(entities, context)
            elif task_type == TaskType.SYSTEM_INFO:
                result = await self._execute_system_info_task(entities, context)
            else:
                result = {'success': False, 'error': f'Task type {task_type} not implemented'}
            
            step.output_data = result
            step.success = result.get('success', False)
            
        except Exception as e:
            step.errors.append(str(e))
            step.success = False
            step.output_data = {'error': str(e)}
        
        step.execution_time = time.time() - start_time
        return step
    
    async def _execute_terminal_task(self, entities: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Execute terminal command task."""
        # Determine command from entities or context
        commands = entities.get('commands', [])
        user_request = context.user_request.lower()
        
        if 'date' in user_request and 'time' in user_request:
            command = self.terminal_executor.universal_commands['date_time'][self.terminal_executor.platform]
        elif 'current directory' in user_request or 'pwd' in user_request:
            command = self.terminal_executor.universal_commands['current_directory'][self.terminal_executor.platform]
        elif 'list files' in user_request or 'ls' in user_request:
            command = self.terminal_executor.universal_commands['list_files'][self.terminal_executor.platform]
        elif commands:
            command = commands[0]
        else:
            # Extract command from user request
            command = self._extract_command_from_request(context.user_request)
        
        if command:
            result = await self.terminal_executor.execute_command(command)
            return {
                'success': result.exit_code == 0,
                'command': result.command,
                'exit_code': result.exit_code,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': result.execution_time
            }
        else:
            return {'success': False, 'error': 'No valid command found'}
    
    async def _execute_file_task(self, entities: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Execute file operation task."""
        files = entities.get('files', [])
        user_request = context.user_request.lower()
        
        if 'create' in user_request and files:
            file_path = files[0]
            operation = await self.file_manager.create_file(file_path, "# New file created by CODY\n")
            return {'success': True, 'operation': 'create', 'source_path': file_path}
        elif 'read' in user_request and files:
            file_path = files[0]
            try:
                content, operation = await self.file_manager.read_file(file_path)
                return {'success': True, 'operation': 'read', 'source_path': file_path, 'content': content}
            except FileNotFoundError:
                return {'success': False, 'error': f'File not found: {file_path}'}
        else:
            return {'success': False, 'error': 'File operation not supported or no file specified'}
    
    async def _execute_system_info_task(self, entities: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Execute system information task."""
        command = self.terminal_executor.universal_commands['system_info'][self.terminal_executor.platform]
        result = await self.terminal_executor.execute_command(command)
        
        return {
            'success': result.exit_code == 0,
            'system_info': result.stdout,
            'command': result.command,
            'execution_time': result.execution_time
        }
    
    def _extract_command_from_request(self, user_request: str) -> str:
        """Extract command from natural language request."""
        request_lower = user_request.lower()
        
        # Common patterns
        if 'show' in request_lower and ('date' in request_lower or 'time' in request_lower):
            return self.terminal_executor.universal_commands['date_time'][self.terminal_executor.platform]
        elif 'current directory' in request_lower or 'where am i' in request_lower:
            return self.terminal_executor.universal_commands['current_directory'][self.terminal_executor.platform]
        elif 'list files' in request_lower or 'show files' in request_lower:
            return self.terminal_executor.universal_commands['list_files'][self.terminal_executor.platform]
        
        # Try to extract direct commands
        command_patterns = [
            r'run\s+["\']?([^"\']+)["\']?',
            r'execute\s+["\']?([^"\']+)["\']?',
            r'command\s+["\']?([^"\']+)["\']?'
        ]
        
        for pattern in command_patterns:
            match = re.search(pattern, request_lower)
            if match:
                return match.group(1)
        
        return ""
