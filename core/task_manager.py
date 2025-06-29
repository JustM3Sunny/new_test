#!/usr/bin/env python3

"""
Multi-threaded Task Manager for CODY Agent
Handles concurrent task execution, predictive prefetching, and performance optimization
"""

import asyncio
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger('CODY.TaskManager')

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Represents a task to be executed."""
    task_id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class PredictiveCache:
    """Cache for predictive prefetching."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_count = defaultdict(int)
        self.access_history = deque(maxlen=max_size)
        self.max_size = max_size
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Any:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                self.access_count[key] += 1
                self.access_history.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = {
                'value': value,
                'created_at': time.time(),
                'ttl': ttl,
                'access_count': 0
            }
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_history:
            # Fallback: remove oldest item
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['created_at'])
            del self.cache[oldest_key]
        else:
            # Remove least recently used
            lru_key = self.access_history.popleft()
            if lru_key in self.cache:
                del self.cache[lru_key]
    
    def predict_next_access(self) -> List[str]:
        """Predict next likely cache accesses."""
        # Simple prediction based on access patterns
        recent_accesses = list(self.access_history)[-10:]
        return list(set(recent_accesses))  # Return unique recent accesses

class MultiThreadedTaskManager:
    """Advanced task manager with multi-threading and predictive capabilities."""
    
    def __init__(self, max_workers: int = 4, enable_prediction: bool = True):
        self.max_workers = max_workers
        self.enable_prediction = enable_prediction
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: Dict[str, Future] = {}
        
        # Caching and prediction
        self.cache = PredictiveCache()
        self.prediction_patterns = defaultdict(list)
        
        # Performance metrics
        self.metrics = {
            'tasks_executed': 0,
            'tasks_failed': 0,
            'average_execution_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Control
        self.running = True
        self.lock = threading.RLock()
        
        # Start background workers
        self._start_background_workers()
    
    def _start_background_workers(self) -> None:
        """Start background worker threads."""
        # Task processor
        self.task_processor_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.task_processor_thread.start()
        
        # Predictive prefetcher
        if self.enable_prediction:
            self.prefetch_thread = threading.Thread(target=self._predictive_prefetcher, daemon=True)
            self.prefetch_thread.start()
        
        # Metrics collector
        self.metrics_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self.metrics_thread.start()
    
    def submit_task(self, task_id: str, function: Callable, *args, 
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   timeout: Optional[float] = None,
                   dependencies: List[str] = None,
                   **kwargs) -> str:
        """
        Submit a task for execution.
        
        Args:
            task_id: Unique identifier for the task
            function: Function to execute
            *args: Function arguments
            priority: Task priority
            timeout: Execution timeout
            dependencies: List of task IDs this task depends on
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task = Task(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            dependencies=dependencies or []
        )
        
        with self.lock:
            self.tasks[task_id] = task
            
            # Check if dependencies are satisfied
            if self._dependencies_satisfied(task):
                self.task_queue.put((priority.value, time.time(), task_id))
        
        logger.info(f"Task {task_id} submitted with priority {priority.name}")
        return task_id
    
    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if self.completed_tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _process_tasks(self) -> None:
        """Background task processor."""
        while self.running:
            try:
                # Get next task from queue (blocks until available)
                priority, timestamp, task_id = self.task_queue.get(timeout=1.0)
                
                with self.lock:
                    if task_id not in self.tasks:
                        continue
                    
                    task = self.tasks[task_id]
                    
                    # Double-check dependencies
                    if not self._dependencies_satisfied(task):
                        # Re-queue the task
                        self.task_queue.put((priority, timestamp, task_id))
                        continue
                    
                    # Update task status
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
                
                # Execute task
                self._execute_task(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
    
    def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(task)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                self.metrics['cache_hits'] += 1
                self._complete_task(task, cached_result, from_cache=True)
                return
            
            self.metrics['cache_misses'] += 1
            
            # Submit to thread pool
            future = self.executor.submit(self._run_task_function, task)
            self.futures[task.task_id] = future
            
            # Handle completion
            future.add_done_callback(lambda f: self._handle_task_completion(task, f))
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            self._fail_task(task, e)
    
    def _run_task_function(self, task: Task) -> Any:
        """Run the actual task function."""
        try:
            return task.function(*task.args, **task.kwargs)
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            raise
    
    def _handle_task_completion(self, task: Task, future: Future) -> None:
        """Handle task completion."""
        try:
            result = future.result(timeout=task.timeout)
            
            # Cache result if appropriate
            cache_key = self._generate_cache_key(task)
            if self._should_cache_result(task, result):
                self.cache.put(cache_key, result)
            
            self._complete_task(task, result)
            
        except Exception as e:
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                self.task_queue.put((task.priority.value, time.time(), task.task_id))
            else:
                self._fail_task(task, e)
    
    def _complete_task(self, task: Task, result: Any, from_cache: bool = False) -> None:
        """Mark task as completed."""
        with self.lock:
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            
            execution_time = (task.completed_at - task.started_at) if task.started_at else 0.0
            
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                metadata={'from_cache': from_cache}
            )
            
            self.completed_tasks[task.task_id] = task_result
            self.metrics['tasks_executed'] += 1
            
            # Update average execution time
            current_avg = self.metrics['average_execution_time']
            task_count = self.metrics['tasks_executed']
            self.metrics['average_execution_time'] = (current_avg * (task_count - 1) + execution_time) / task_count
            
            # Check for dependent tasks
            self._check_dependent_tasks(task.task_id)
        
        logger.info(f"Task {task.task_id} completed in {execution_time:.2f}s")
    
    def _fail_task(self, task: Task, error: Exception) -> None:
        """Mark task as failed."""
        with self.lock:
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            task.error = error
            
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=error
            )
            
            self.completed_tasks[task.task_id] = task_result
            self.metrics['tasks_failed'] += 1
        
        logger.error(f"Task {task.task_id} failed: {error}")
    
    def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """Check if any pending tasks can now be executed."""
        for task_id, task in self.tasks.items():
            if (task.status == TaskStatus.PENDING and 
                completed_task_id in task.dependencies and
                self._dependencies_satisfied(task)):
                
                self.task_queue.put((task.priority.value, time.time(), task_id))
    
    def _generate_cache_key(self, task: Task) -> str:
        """Generate cache key for task."""
        import hashlib
        key_data = f"{task.function.__name__}:{str(task.args)}:{str(sorted(task.kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _should_cache_result(self, task: Task, result: Any) -> bool:
        """Determine if task result should be cached."""
        # Cache results for read-only operations
        function_name = task.function.__name__
        cacheable_functions = ['read_file', 'analyze_code', 'search_code', 'web_search']
        
        return any(func in function_name for func in cacheable_functions)
    
    def _predictive_prefetcher(self) -> None:
        """Background predictive prefetching."""
        while self.running:
            try:
                time.sleep(5)  # Run every 5 seconds
                
                # Get prediction patterns
                predicted_keys = self.cache.predict_next_access()
                
                # Prefetch likely needed data
                for key in predicted_keys[:3]:  # Limit prefetching
                    if key not in self.cache.cache:
                        # This would trigger prefetching logic
                        pass
                
            except Exception as e:
                logger.error(f"Error in predictive prefetcher: {e}")
    
    def _collect_metrics(self) -> None:
        """Background metrics collection."""
        while self.running:
            try:
                time.sleep(60)  # Collect every 60 seconds instead of 10

                # Log performance metrics only if there's activity
                if self.metrics['tasks_executed'] > 0:
                    logger.debug(f"Performance metrics: {self.metrics}")  # Changed to debug
                
                # Clean up old completed tasks
                current_time = time.time()
                old_tasks = [
                    task_id for task_id, result in self.completed_tasks.items()
                    if current_time - result.metadata.get('completed_at', 0) > 3600  # 1 hour
                ]
                
                for task_id in old_tasks:
                    del self.completed_tasks[task_id]
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        with self.lock:
            if task_id in self.tasks:
                return self.tasks[task_id].status
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id].status
            return None
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed task."""
        with self.lock:
            return self.completed_tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    task.status = TaskStatus.CANCELLED
                    
                    # Cancel future if running
                    if task_id in self.futures:
                        self.futures[task_id].cancel()
                        del self.futures[task_id]
                    
                    return True
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            return self.metrics.copy()
    
    def shutdown(self) -> None:
        """Shutdown the task manager."""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Task manager shutdown complete")
