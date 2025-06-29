#!/usr/bin/env python3

"""
Performance Core Module for CODY Agent
Ultra-low latency, parallelism, prefetching, and real-time optimizations
"""

import asyncio
import threading
import time
import queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import psutil
import gc
import weakref
from collections import defaultdict, deque
import pickle
import hashlib

logger = logging.getLogger('CODY.PerformanceCore')

class TaskPriority(Enum):
    """Task priority levels for performance optimization."""
    CRITICAL = 0    # User-facing, immediate response
    HIGH = 1        # Important background tasks
    MEDIUM = 2      # Standard operations
    LOW = 3         # Cleanup, optimization
    BACKGROUND = 4  # Prefetching, caching

class PerformanceMetric(Enum):
    """Performance metrics to track."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    TASK_COMPLETION_TIME = "task_completion_time"

@dataclass
class PerformanceTask:
    """High-performance task wrapper."""
    task_id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    estimated_duration: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None

@dataclass
class PerformanceStats:
    """Performance statistics tracking."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_latency: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_utilization: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    start_time: float = field(default_factory=time.time)

class UltraFastCache:
    """Ultra-fast caching system with intelligent eviction."""
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.size_tracker = {}
        self.current_size = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Any:
        """Ultra-fast cache retrieval."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Ultra-fast cache storage with intelligent eviction."""
        try:
            # Estimate size
            serialized = pickle.dumps(value)
            size = len(serialized)
            
            with self.lock:
                # Check if we need to evict
                if self.current_size + size > self.max_size_bytes:
                    self._evict_lru(size)
                
                # Store the value
                self.cache[key] = value
                self.size_tracker[key] = size
                self.access_times[key] = time.time()
                self.access_counts[key] = 1
                self.current_size += size
                
                return True
        except Exception as e:
            logger.warning(f"Cache put failed: {e}")
            return False
    
    def _evict_lru(self, needed_size: int) -> None:
        """Evict least recently used items."""
        # Sort by access time and frequency
        candidates = []
        for key in self.cache:
            score = self.access_times[key] + (self.access_counts[key] * 0.1)
            candidates.append((score, key))
        
        candidates.sort()  # Lowest score first (LRU)
        
        freed_size = 0
        for _, key in candidates:
            if freed_size >= needed_size:
                break
            
            freed_size += self.size_tracker.get(key, 0)
            self._remove_key(key)
    
    def _remove_key(self, key: str) -> None:
        """Remove a key from cache."""
        if key in self.cache:
            self.current_size -= self.size_tracker.get(key, 0)
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            del self.size_tracker[key]

class PredictivePrefetcher:
    """Predictive prefetching engine for anticipating user needs."""
    
    def __init__(self, cache: UltraFastCache):
        self.cache = cache
        self.pattern_history = deque(maxlen=1000)
        self.prediction_models = {}
        self.prefetch_queue = queue.Queue()
        self.is_running = True
        
        # Start prefetch worker
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def record_access(self, key: str, context: Dict[str, Any]) -> None:
        """Record access pattern for prediction."""
        self.pattern_history.append({
            'key': key,
            'timestamp': time.time(),
            'context': context
        })
        
        # Trigger prediction
        self._predict_next_access()
    
    def _predict_next_access(self) -> None:
        """Predict next likely access patterns."""
        if len(self.pattern_history) < 3:
            return
        
        # Simple pattern recognition
        recent_patterns = list(self.pattern_history)[-10:]
        
        # Look for sequential patterns
        for i in range(len(recent_patterns) - 2):
            pattern = [p['key'] for p in recent_patterns[i:i+3]]
            if len(set(pattern)) == 3:  # All different keys
                # Predict next in sequence
                predicted_key = self._generate_predicted_key(pattern)
                if predicted_key:
                    self.prefetch_queue.put(predicted_key)
    
    def _generate_predicted_key(self, pattern: List[str]) -> Optional[str]:
        """Generate predicted key based on pattern."""
        # Simple heuristic: if pattern is file1.py, file2.py, predict file3.py
        if all('.py' in key for key in pattern):
            numbers = []
            for key in pattern:
                import re
                match = re.search(r'(\d+)', key)
                if match:
                    numbers.append(int(match.group(1)))
            
            if len(numbers) == 3 and numbers[1] == numbers[0] + 1 and numbers[2] == numbers[1] + 1:
                next_num = numbers[2] + 1
                predicted = pattern[-1].replace(str(numbers[2]), str(next_num))
                return predicted
        
        return None
    
    def _prefetch_worker(self) -> None:
        """Background worker for prefetching."""
        while self.is_running:
            try:
                predicted_key = self.prefetch_queue.get(timeout=1.0)
                
                # Check if already cached
                if self.cache.get(predicted_key) is None:
                    # Simulate prefetching (in real implementation, this would load the data)
                    logger.debug(f"Prefetching: {predicted_key}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Prefetch error: {e}")

class PerformanceMonitor:
    """Real-time performance monitoring and optimization."""
    
    def __init__(self):
        self.stats = PerformanceStats()
        self.metrics_history = defaultdict(deque)
        self.alert_thresholds = {
            PerformanceMetric.LATENCY: 1.0,  # 1 second
            PerformanceMetric.MEMORY_USAGE: 80.0,  # 80% of available
            PerformanceMetric.CPU_USAGE: 90.0,  # 90% CPU
        }
        self.monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def record_task_completion(self, task: PerformanceTask, duration: float, success: bool) -> None:
        """Record task completion metrics."""
        self.stats.total_tasks += 1
        
        if success:
            self.stats.completed_tasks += 1
        else:
            self.stats.failed_tasks += 1
        
        # Update average latency
        total_completed = self.stats.completed_tasks
        if total_completed > 0:
            self.stats.average_latency = (
                (self.stats.average_latency * (total_completed - 1) + duration) / total_completed
            )
        
        # Record in history
        self.metrics_history[PerformanceMetric.LATENCY].append(duration)
        self.metrics_history[PerformanceMetric.TASK_COMPLETION_TIME].append(time.time())
        
        # Keep only recent history
        for metric_history in self.metrics_history.values():
            if len(metric_history) > 1000:
                metric_history.popleft()
    
    def _monitor_loop(self) -> None:
        """Continuous monitoring loop."""
        while self.monitoring:
            try:
                # Monitor system resources
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)
                
                self.stats.cpu_utilization = cpu_percent
                
                # Update peak memory
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                if memory_mb > self.stats.peak_memory_mb:
                    self.stats.peak_memory_mb = memory_mb
                
                # Record metrics
                self.metrics_history[PerformanceMetric.MEMORY_USAGE].append(memory_percent)
                self.metrics_history[PerformanceMetric.CPU_USAGE].append(cpu_percent)

                # Check for alerts
                self._check_alerts(memory_percent, cpu_percent)

                time.sleep(10)  # Monitor every 10 seconds instead of 1
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _check_alerts(self, memory_percent: float, cpu_percent: float) -> None:
        """Check for performance alerts."""
        # Only alert every 30 seconds to reduce spam
        current_time = time.time()
        if not hasattr(self, '_last_alert_time'):
            self._last_alert_time = 0

        if current_time - self._last_alert_time > 30:  # 30 seconds cooldown
            if memory_percent > self.alert_thresholds[PerformanceMetric.MEMORY_USAGE]:
                logger.debug(f"High memory usage: {memory_percent:.1f}%")  # Changed to debug
                self._trigger_memory_optimization()
                self._last_alert_time = current_time

            if cpu_percent > self.alert_thresholds[PerformanceMetric.CPU_USAGE]:
                logger.debug(f"High CPU usage: {cpu_percent:.1f}%")  # Changed to debug
                self._last_alert_time = current_time
    
    def _trigger_memory_optimization(self) -> None:
        """Trigger memory optimization."""
        gc.collect()  # Force garbage collection
        logger.debug("Triggered memory optimization")  # Changed to debug level

class UltraPerformanceCore:
    """Main ultra-performance core system."""
    
    def __init__(self, max_workers: int = None, cache_size_mb: int = 100):
        # Determine optimal worker count
        if max_workers is None:
            max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
        
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers // 2)
        
        # Performance components
        self.cache = UltraFastCache(cache_size_mb)
        self.prefetcher = PredictivePrefetcher(self.cache)
        self.monitor = PerformanceMonitor()
        
        # Task management
        self.priority_queues = {priority: queue.PriorityQueue() for priority in TaskPriority}
        self.active_tasks = {}
        self.task_results = {}
        
        # Performance optimization
        self.batch_size = 10
        self.batch_timeout = 0.1  # 100ms
        
        # Start task dispatcher
        self.dispatcher_thread = threading.Thread(target=self._task_dispatcher, daemon=True)
        self.dispatcher_thread.start()
    
    async def execute_task(self, task: PerformanceTask) -> Any:
        """Execute a task with ultra-high performance."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(task)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                self.monitor.stats.cache_hits += 1
                self.prefetcher.record_access(cache_key, {'task_type': 'cached'})
                return cached_result
            
            self.monitor.stats.cache_misses += 1
            
            # Execute task based on priority and type
            if task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                # Execute immediately in thread pool
                future = self.thread_pool.submit(self._execute_function, task)
                result = future.result(timeout=task.timeout)
            else:
                # Queue for batch processing
                self.priority_queues[task.priority].put((time.time(), task))
                
                # Wait for result
                result = await self._wait_for_result(task.task_id, task.timeout or 30.0)
            
            # Cache result if appropriate
            if self._should_cache(task, result):
                self.cache.put(cache_key, result)
                self.prefetcher.record_access(cache_key, {'task_type': 'computed'})
            
            # Record performance metrics
            duration = time.time() - start_time
            self.monitor.record_task_completion(task, duration, True)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_task_completion(task, duration, False)
            logger.error(f"Task execution failed: {e}")
            raise
    
    def _generate_cache_key(self, task: PerformanceTask) -> str:
        """Generate cache key for task."""
        key_data = f"{task.function.__name__}:{str(task.args)}:{str(sorted(task.kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _should_cache(self, task: PerformanceTask, result: Any) -> bool:
        """Determine if result should be cached."""
        # Cache read-only operations and expensive computations
        function_name = task.function.__name__
        cacheable_functions = [
            'read_file', 'analyze_code', 'search_code', 'web_search',
            'parse_ast', 'lint_code', 'format_code'
        ]
        
        return any(func in function_name for func in cacheable_functions)
    
    def _execute_function(self, task: PerformanceTask) -> Any:
        """Execute the actual function."""
        return task.function(*task.args, **task.kwargs)
    
    async def _wait_for_result(self, task_id: str, timeout: float) -> Any:
        """Wait for task result with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.task_results:
                result = self.task_results.pop(task_id)
                if isinstance(result, Exception):
                    raise result
                return result
            
            await asyncio.sleep(0.01)  # 10ms polling
        
        raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
    
    def _task_dispatcher(self) -> None:
        """Background task dispatcher for batch processing."""
        while True:
            try:
                # Process tasks by priority
                for priority in TaskPriority:
                    if not self.priority_queues[priority].empty():
                        self._process_priority_queue(priority)
                
                time.sleep(0.01)  # 10ms dispatch cycle
                
            except Exception as e:
                logger.error(f"Dispatcher error: {e}")
    
    def _process_priority_queue(self, priority: TaskPriority) -> None:
        """Process tasks from a priority queue."""
        queue_obj = self.priority_queues[priority]
        batch = []
        
        # Collect batch
        start_time = time.time()
        while (len(batch) < self.batch_size and 
               time.time() - start_time < self.batch_timeout and
               not queue_obj.empty()):
            
            try:
                _, task = queue_obj.get_nowait()
                batch.append(task)
            except queue.Empty:
                break
        
        # Execute batch
        if batch:
            self._execute_batch(batch)
    
    def _execute_batch(self, tasks: List[PerformanceTask]) -> None:
        """Execute a batch of tasks."""
        futures = []
        
        for task in tasks:
            future = self.thread_pool.submit(self._execute_function, task)
            futures.append((task.task_id, future))
        
        # Collect results
        for task_id, future in futures:
            try:
                result = future.result(timeout=30.0)
                self.task_results[task_id] = result
            except Exception as e:
                self.task_results[task_id] = e
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'stats': self.monitor.stats.__dict__,
            'cache_stats': {
                'size_mb': self.cache.current_size / (1024 * 1024),
                'hit_rate': (self.monitor.stats.cache_hits / 
                           max(self.monitor.stats.cache_hits + self.monitor.stats.cache_misses, 1)),
                'total_items': len(self.cache.cache)
            },
            'system_stats': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'active_threads': threading.active_count()
            },
            'queue_stats': {
                priority.name: self.priority_queues[priority].qsize() 
                for priority in TaskPriority
            }
        }
    
    def optimize_performance(self) -> None:
        """Trigger performance optimizations."""
        # Clear old cache entries
        current_time = time.time()
        old_keys = []
        
        for key, access_time in self.cache.access_times.items():
            if current_time - access_time > 3600:  # 1 hour old
                old_keys.append(key)
        
        for key in old_keys:
            self.cache._remove_key(key)
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Performance optimization completed")
    
    def shutdown(self) -> None:
        """Shutdown performance core."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        self.monitor.monitoring = False
        self.prefetcher.is_running = False
        logger.info("Performance core shutdown completed")
