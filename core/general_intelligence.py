#!/usr/bin/env python3

"""
General Intelligence Module for CODY Agent
Implements advanced reasoning, planning, and decision-making capabilities
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger('CODY.GeneralIntelligence')

class ReasoningType(Enum):
    """Types of reasoning approaches."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STEP_BY_STEP = "step_by_step"
    PROBLEM_DECOMPOSITION = "problem_decomposition"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"

class PlanningStrategy(Enum):
    """Planning strategies for task execution."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"

@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_id: int
    description: str
    reasoning_type: ReasoningType
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class Plan:
    """A plan for executing a complex task."""
    plan_id: str
    goal: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    strategy: PlanningStrategy = PlanningStrategy.SEQUENTIAL
    estimated_time: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

class AdvancedReasoner:
    """Advanced reasoning engine with multiple reasoning strategies."""
    
    def __init__(self):
        self.reasoning_history = []
        self.knowledge_base = {}
        self.pattern_library = {}
        
    def chain_of_thought_reasoning(self, problem: str, context: Dict[str, Any]) -> List[ReasoningStep]:
        """Implement Chain-of-Thought reasoning for complex problems."""
        steps = []
        step_counter = 0
        
        # Step 1: Problem Understanding
        step_counter += 1
        understanding_step = ReasoningStep(
            step_id=step_counter,
            description=f"Understanding the problem: {problem}",
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            input_data={"problem": problem, "context": context},
            confidence=0.9
        )
        
        # Analyze problem complexity
        complexity_indicators = self._analyze_complexity(problem)
        understanding_step.output_data = {
            "complexity": complexity_indicators,
            "key_concepts": self._extract_key_concepts(problem),
            "domain": self._identify_domain(problem)
        }
        steps.append(understanding_step)
        
        # Step 2: Context Analysis
        step_counter += 1
        context_step = ReasoningStep(
            step_id=step_counter,
            description="Analyzing available context and resources",
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            input_data=context,
            confidence=0.8
        )
        
        context_analysis = self._analyze_context(context)
        context_step.output_data = context_analysis
        steps.append(context_step)
        
        # Step 3: Solution Strategy
        step_counter += 1
        strategy_step = ReasoningStep(
            step_id=step_counter,
            description="Determining optimal solution strategy",
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            confidence=0.85
        )
        
        strategy = self._determine_strategy(problem, complexity_indicators, context_analysis)
        strategy_step.output_data = strategy
        steps.append(strategy_step)
        
        # Step 4: Implementation Planning
        step_counter += 1
        planning_step = ReasoningStep(
            step_id=step_counter,
            description="Creating detailed implementation plan",
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            confidence=0.8
        )
        
        implementation_plan = self._create_implementation_plan(strategy, context)
        planning_step.output_data = implementation_plan
        steps.append(planning_step)
        
        return steps
    
    def _analyze_complexity(self, problem: str) -> Dict[str, Any]:
        """Analyze the complexity of a given problem."""
        complexity_indicators = {
            "length": len(problem),
            "technical_terms": len(re.findall(r'\b(?:function|class|method|variable|API|database|server)\b', problem.lower())),
            "file_operations": len(re.findall(r'\b(?:create|edit|delete|read|write|file|folder)\b', problem.lower())),
            "programming_languages": len(re.findall(r'\b(?:python|javascript|java|cpp|go|rust|typescript)\b', problem.lower())),
            "complexity_keywords": len(re.findall(r'\b(?:complex|advanced|sophisticated|comprehensive|full)\b', problem.lower()))
        }
        
        # Calculate overall complexity score
        total_score = sum(complexity_indicators.values())
        if total_score > 15:
            complexity_level = "high"
        elif total_score > 8:
            complexity_level = "medium"
        else:
            complexity_level = "low"
        
        complexity_indicators["level"] = complexity_level
        complexity_indicators["score"] = total_score
        
        return complexity_indicators
    
    def _extract_key_concepts(self, problem: str) -> List[str]:
        """Extract key concepts from the problem statement."""
        # Common programming concepts
        concepts = []
        
        concept_patterns = {
            "authentication": r'\b(?:login|auth|authentication|password|user)\b',
            "database": r'\b(?:database|db|sql|mongodb|postgres)\b',
            "api": r'\b(?:api|rest|endpoint|request|response)\b',
            "frontend": r'\b(?:ui|frontend|react|vue|angular|html|css)\b',
            "backend": r'\b(?:backend|server|node|express|flask|django)\b',
            "testing": r'\b(?:test|testing|unit|integration|pytest)\b',
            "deployment": r'\b(?:deploy|deployment|docker|kubernetes|aws)\b'
        }
        
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, problem.lower()):
                concepts.append(concept)
        
        return concepts
    
    def _identify_domain(self, problem: str) -> str:
        """Identify the primary domain of the problem."""
        domain_keywords = {
            "web_development": ["web", "website", "html", "css", "javascript", "react", "vue", "angular"],
            "backend_development": ["server", "api", "database", "backend", "node", "python", "flask", "django"],
            "mobile_development": ["mobile", "app", "android", "ios", "react native", "flutter"],
            "data_science": ["data", "analysis", "machine learning", "ai", "pandas", "numpy"],
            "devops": ["deploy", "docker", "kubernetes", "ci/cd", "aws", "cloud"],
            "desktop_development": ["desktop", "gui", "tkinter", "qt", "electron"]
        }
        
        problem_lower = problem.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in problem_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return "general_programming"
    
    def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the available context for decision making."""
        analysis = {
            "available_files": context.get("open_files", []),
            "file_count": len(context.get("open_files", [])),
            "has_terminal_history": bool(context.get("terminal_output", [])),
            "has_chat_history": bool(context.get("chat_history", [])),
            "working_directory": context.get("current_directory", "."),
            "project_type": self._infer_project_type(context.get("open_files", []))
        }
        
        return analysis
    
    def _infer_project_type(self, files: List[str]) -> str:
        """Infer project type from file extensions."""
        extensions = [file.split('.')[-1].lower() for file in files if '.' in file]
        
        if any(ext in extensions for ext in ['py']):
            return "python_project"
        elif any(ext in extensions for ext in ['js', 'jsx', 'ts', 'tsx']):
            return "javascript_project"
        elif any(ext in extensions for ext in ['java']):
            return "java_project"
        elif any(ext in extensions for ext in ['cpp', 'c', 'h']):
            return "cpp_project"
        elif any(ext in extensions for ext in ['go']):
            return "go_project"
        
        return "mixed_project"
    
    def _determine_strategy(self, problem: str, complexity: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the optimal strategy for solving the problem."""
        strategy = {
            "approach": "incremental",
            "priority": "functionality_first",
            "testing_strategy": "test_driven",
            "deployment_strategy": "local_first"
        }
        
        # Adjust strategy based on complexity
        if complexity["level"] == "high":
            strategy["approach"] = "modular_decomposition"
            strategy["testing_strategy"] = "comprehensive"
        elif complexity["level"] == "low":
            strategy["approach"] = "direct_implementation"
            strategy["testing_strategy"] = "basic"
        
        # Adjust based on context
        if context.get("has_terminal_history"):
            strategy["build_system"] = "existing_tools"
        else:
            strategy["build_system"] = "setup_required"
        
        return strategy
    
    def _create_implementation_plan(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed implementation plan."""
        plan = {
            "phases": [],
            "estimated_steps": 0,
            "risk_factors": [],
            "success_metrics": []
        }
        
        # Phase 1: Setup and Preparation
        plan["phases"].append({
            "name": "setup",
            "description": "Environment setup and preparation",
            "tasks": ["verify_dependencies", "setup_project_structure", "initialize_git"]
        })
        
        # Phase 2: Core Implementation
        plan["phases"].append({
            "name": "implementation",
            "description": "Core functionality implementation",
            "tasks": ["create_main_components", "implement_core_logic", "add_error_handling"]
        })
        
        # Phase 3: Testing and Validation
        plan["phases"].append({
            "name": "testing",
            "description": "Testing and quality assurance",
            "tasks": ["write_unit_tests", "integration_testing", "code_review"]
        })
        
        # Phase 4: Finalization
        plan["phases"].append({
            "name": "finalization",
            "description": "Documentation and deployment preparation",
            "tasks": ["add_documentation", "optimize_performance", "prepare_deployment"]
        })
        
        plan["estimated_steps"] = sum(len(phase["tasks"]) for phase in plan["phases"])
        
        return plan

class IntelligentPlanner:
    """Intelligent planning system for complex task execution."""
    
    def __init__(self):
        self.plans = {}
        self.execution_history = []
        
    def create_plan(self, goal: str, context: Dict[str, Any], reasoning_steps: List[ReasoningStep]) -> Plan:
        """Create a comprehensive plan for achieving a goal."""
        plan_id = f"plan_{int(time.time())}"
        
        # Extract implementation plan from reasoning
        implementation_data = None
        for step in reasoning_steps:
            if "implementation_plan" in step.output_data:
                implementation_data = step.output_data["implementation_plan"]
                break
        
        if not implementation_data:
            implementation_data = {"phases": [], "estimated_steps": 1}
        
        plan = Plan(
            plan_id=plan_id,
            goal=goal,
            strategy=self._determine_strategy(goal, context),
            estimated_time=self._estimate_time(implementation_data),
            success_criteria=self._define_success_criteria(goal)
        )
        
        # Convert phases to executable steps
        for phase in implementation_data.get("phases", []):
            for task in phase.get("tasks", []):
                plan.steps.append({
                    "id": len(plan.steps) + 1,
                    "phase": phase["name"],
                    "task": task,
                    "status": "pending",
                    "dependencies": [],
                    "estimated_time": 60  # seconds
                })
        
        self.plans[plan_id] = plan
        return plan
    
    def _determine_strategy(self, goal: str, context: Dict[str, Any]) -> PlanningStrategy:
        """Determine the best planning strategy."""
        if "parallel" in goal.lower() or len(context.get("open_files", [])) > 5:
            return PlanningStrategy.PARALLEL
        elif "complex" in goal.lower() or "comprehensive" in goal.lower():
            return PlanningStrategy.HIERARCHICAL
        else:
            return PlanningStrategy.SEQUENTIAL
    
    def _estimate_time(self, implementation_data: Dict[str, Any]) -> float:
        """Estimate time required for plan execution."""
        base_time = 300  # 5 minutes base
        steps = implementation_data.get("estimated_steps", 1)
        return base_time + (steps * 60)  # 1 minute per step
    
    def _define_success_criteria(self, goal: str) -> List[str]:
        """Define success criteria for the goal."""
        criteria = ["Task completed without errors"]
        
        if "test" in goal.lower():
            criteria.append("All tests pass")
        
        if "deploy" in goal.lower():
            criteria.append("Successfully deployed")
        
        if "create" in goal.lower():
            criteria.append("Files created and functional")
        
        return criteria
    
    def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute a plan step by step."""
        if plan_id not in self.plans:
            return {"error": "Plan not found"}
        
        plan = self.plans[plan_id]
        execution_result = {
            "plan_id": plan_id,
            "status": "in_progress",
            "completed_steps": 0,
            "total_steps": len(plan.steps),
            "errors": []
        }
        
        # Execute steps based on strategy
        if plan.strategy == PlanningStrategy.SEQUENTIAL:
            execution_result = self._execute_sequential(plan, execution_result)
        elif plan.strategy == PlanningStrategy.PARALLEL:
            execution_result = self._execute_parallel(plan, execution_result)
        else:
            execution_result = self._execute_adaptive(plan, execution_result)
        
        return execution_result
    
    def _execute_sequential(self, plan: Plan, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan steps sequentially."""
        for step in plan.steps:
            try:
                # Simulate step execution
                step["status"] = "completed"
                result["completed_steps"] += 1
                logger.info(f"Completed step: {step['task']}")
            except Exception as e:
                step["status"] = "failed"
                result["errors"].append(f"Step {step['id']} failed: {str(e)}")
        
        result["status"] = "completed" if not result["errors"] else "failed"
        return result
    
    def _execute_parallel(self, plan: Plan, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan steps in parallel where possible."""
        # For now, simulate parallel execution
        return self._execute_sequential(plan, result)
    
    def _execute_adaptive(self, plan: Plan, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan with adaptive strategy."""
        return self._execute_sequential(plan, result)

class GeneralIntelligence:
    """Main General Intelligence system combining reasoning and planning."""
    
    def __init__(self):
        self.reasoner = AdvancedReasoner()
        self.planner = IntelligentPlanner()
        self.memory = {}
        self.learning_data = []
    
    async def process_complex_request(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complex request using advanced reasoning and planning."""
        start_time = time.time()
        
        # Step 1: Advanced Reasoning
        reasoning_steps = self.reasoner.chain_of_thought_reasoning(request, context)
        
        # Step 2: Create Execution Plan
        plan = self.planner.create_plan(request, context, reasoning_steps)
        
        # Step 3: Execute Plan
        execution_result = self.planner.execute_plan(plan.plan_id)
        
        # Step 4: Learn from Experience
        self._learn_from_execution(request, reasoning_steps, execution_result)
        
        processing_time = time.time() - start_time
        
        return {
            "request": request,
            "reasoning_steps": [step.__dict__ for step in reasoning_steps],
            "plan": plan.__dict__,
            "execution_result": execution_result,
            "processing_time": processing_time,
            "success": execution_result.get("status") == "completed"
        }
    
    def _learn_from_execution(self, request: str, reasoning_steps: List[ReasoningStep], 
                            execution_result: Dict[str, Any]) -> None:
        """Learn from execution results to improve future performance."""
        learning_entry = {
            "timestamp": time.time(),
            "request": request,
            "reasoning_quality": sum(step.confidence for step in reasoning_steps) / len(reasoning_steps),
            "execution_success": execution_result.get("status") == "completed",
            "errors": execution_result.get("errors", []),
            "completion_rate": execution_result.get("completed_steps", 0) / max(execution_result.get("total_steps", 1), 1)
        }
        
        self.learning_data.append(learning_entry)
        
        # Update memory with successful patterns
        if learning_entry["execution_success"]:
            request_type = self._categorize_request(request)
            if request_type not in self.memory:
                self.memory[request_type] = []
            
            self.memory[request_type].append({
                "request": request,
                "reasoning_pattern": [step.description for step in reasoning_steps],
                "success_rate": 1.0
            })
    
    def _categorize_request(self, request: str) -> str:
        """Categorize request for memory storage."""
        if any(word in request.lower() for word in ["create", "generate", "build"]):
            return "creation"
        elif any(word in request.lower() for word in ["fix", "debug", "error"]):
            return "debugging"
        elif any(word in request.lower() for word in ["search", "find", "look"]):
            return "search"
        else:
            return "general"
    
    def get_intelligence_metrics(self) -> Dict[str, Any]:
        """Get metrics about the intelligence system performance."""
        if not self.learning_data:
            return {"status": "no_data"}
        
        recent_data = self.learning_data[-10:]  # Last 10 executions
        
        return {
            "total_executions": len(self.learning_data),
            "recent_success_rate": sum(1 for entry in recent_data if entry["execution_success"]) / len(recent_data),
            "average_reasoning_quality": sum(entry["reasoning_quality"] for entry in recent_data) / len(recent_data),
            "memory_categories": list(self.memory.keys()),
            "learning_trends": self._analyze_learning_trends()
        }
    
    def _analyze_learning_trends(self) -> Dict[str, Any]:
        """Analyze learning trends over time."""
        if len(self.learning_data) < 5:
            return {"status": "insufficient_data"}
        
        recent_success = sum(1 for entry in self.learning_data[-5:] if entry["execution_success"])
        older_success = sum(1 for entry in self.learning_data[-10:-5] if entry["execution_success"])
        
        return {
            "recent_success_rate": recent_success / 5,
            "previous_success_rate": older_success / 5 if len(self.learning_data) >= 10 else 0,
            "improvement": recent_success / 5 - (older_success / 5 if len(self.learning_data) >= 10 else 0)
        }
