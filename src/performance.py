"""
Advanced Performance Monitoring and Optimization Utilities
High-performance system monitoring with automatic optimization
"""

import logging
import psutil
import gc
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Any
from functools import wraps
from dataclasses import dataclass, field
from contextlib import contextmanager
import warnings

# Configure high-performance logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container"""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    start_memory_mb: float = 0.0
    end_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gc_collections: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
            self.memory_delta_mb = self.end_memory_mb - self.start_memory_mb

@dataclass
class SystemResourceInfo:
    """Current system resource information"""
    cpu_count_physical: int
    cpu_count_logical: int
    cpu_usage_percent: float
    memory_total_gb: float
    memory_available_gb: float
    memory_usage_percent: float
    disk_usage_percent: float
    load_average: Optional[List[float]] = None
    
    @classmethod
    def capture_current(cls) -> 'SystemResourceInfo':
        """Capture current system resource state"""
        cpu_info = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        try:
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        except AttributeError:
            load_avg = None
        
        return cls(
            cpu_count_physical=psutil.cpu_count(logical=False) or psutil.cpu_count(),
            cpu_count_logical=psutil.cpu_count(logical=True),
            cpu_usage_percent=cpu_info,
            memory_total_gb=memory_info.total / (1024**3),
            memory_available_gb=memory_info.available / (1024**3),
            memory_usage_percent=memory_info.percent,
            disk_usage_percent=disk_info.percent,
            load_average=load_avg
        )

class AdvancedPerformanceMonitor:
    """Advanced performance monitoring with real-time optimization"""
    
    def __init__(self, enable_continuous_monitoring: bool = False):
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_operation: Optional[PerformanceMetrics] = None
        self.enable_continuous_monitoring = enable_continuous_monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.peak_memory_tracker = 0.0
        
        # Performance thresholds
        self.slow_operation_threshold = 30.0  # seconds
        self.memory_warning_threshold = 1000.0  # MB
        self.cpu_warning_threshold = 90.0  # percent
        
        if enable_continuous_monitoring:
            self._start_continuous_monitoring()
    
    def start(self, operation_name: str) -> None:
        """Start monitoring an operation with enhanced tracking"""
        if self.current_operation is not None:
            logger.warning(f"⚠️ Starting '{operation_name}' while '{self.current_operation.operation_name}' is still running")
        
        # Force garbage collection before starting
        gc.collect()
        
        # Capture initial state
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.current_operation = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            start_memory_mb=memory_info.rss / (1024 * 1024),
            cpu_usage_percent=psutil.cpu_percent()
        )
        
        self.peak_memory_tracker = self.current_operation.start_memory_mb
        
        logger.info(f"🚀 Starting {operation_name} | Memory: {self.current_operation.start_memory_mb:.1f}MB")
    
    def end(self) -> Optional[PerformanceMetrics]:
        """End monitoring and return comprehensive metrics"""
        if self.current_operation is None:
            logger.warning("⚠️ No operation currently being monitored")
            return None
        
        # Capture final state
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.current_operation.end_time = datetime.now()
        self.current_operation.end_memory_mb = memory_info.rss / (1024 * 1024)
        self.current_operation.peak_memory_mb = self.peak_memory_tracker
        self.current_operation.cpu_usage_percent = psutil.cpu_percent()
        self.current_operation.gc_collections = len(gc.get_stats())
        
        # Calculate derived metrics
        self.current_operation.duration_seconds = (
            self.current_operation.end_time - self.current_operation.start_time
        ).total_seconds()
        self.current_operation.memory_delta_mb = (
            self.current_operation.end_memory_mb - self.current_operation.start_memory_mb
        )
        
        # Log results with performance analysis
        self._log_operation_results(self.current_operation)
        
        # Store in history
        self.metrics_history.append(self.current_operation)
        
        # Return metrics and reset
        result = self.current_operation
        self.current_operation = None
        
        return result
    
    def _log_operation_results(self, metrics: PerformanceMetrics) -> None:
        """Log operation results with intelligent analysis"""
        duration = metrics.duration_seconds
        memory_delta = metrics.memory_delta_mb
        
        # Base log message
        logger.info(f"✅ {metrics.operation_name} completed in {duration:.2f}s")
        logger.info(f"📊 Memory: {memory_delta:+.1f}MB (Peak: {metrics.peak_memory_mb:.1f}MB)")
        
        # Performance warnings
        if duration > self.slow_operation_threshold:
            logger.warning(f"🐌 SLOW OPERATION: {metrics.operation_name} took {duration:.1f}s (threshold: {self.slow_operation_threshold}s)")
            self._suggest_optimizations(metrics)
        
        if abs(memory_delta) > self.memory_warning_threshold:
            logger.warning(f"🧠 HIGH MEMORY USAGE: {metrics.operation_name} used {memory_delta:+.1f}MB")
        
        if metrics.cpu_usage_percent > self.cpu_warning_threshold:
            logger.warning(f"🔥 HIGH CPU USAGE: {metrics.cpu_usage_percent:.1f}%")
    
    def _suggest_optimizations(self, metrics: PerformanceMetrics) -> None:
        """Provide intelligent optimization suggestions"""
        suggestions = []
        
        if metrics.duration_seconds > 60:
            suggestions.append("Consider enabling batch processing or increasing chunk size")
        
        if metrics.memory_delta_mb > 2000:
            suggestions.append("Consider reducing chunk size or enabling memory-efficient mode")
        
        if metrics.cpu_usage_percent > 90:
            suggestions.append("Consider reducing parallel processing cores")
        
        if suggestions:
            logger.info("💡 Optimization suggestions:")
            for suggestion in suggestions:
                logger.info(f"   • {suggestion}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics_history:
            return {"message": "No operations recorded"}
        
        total_operations = len(self.metrics_history)
        total_time = sum(m.duration_seconds for m in self.metrics_history)
        avg_time = total_time / total_operations
        
        memory_changes = [m.memory_delta_mb for m in self.metrics_history]
        avg_memory_delta = sum(memory_changes) / len(memory_changes)
        
        return {
            "total_operations": total_operations,
            "total_time_seconds": total_time,
            "average_time_seconds": avg_time,
            "average_memory_delta_mb": avg_memory_delta,
            "slowest_operation": max(self.metrics_history, key=lambda m: m.duration_seconds).operation_name,
            "most_memory_intensive": max(self.metrics_history, key=lambda m: abs(m.memory_delta_mb)).operation_name
        }
    
    def _start_continuous_monitoring(self) -> None:
        """Start continuous system monitoring in background thread"""
        def monitor():
            while not self.stop_monitoring.is_set():
                if self.current_operation:
                    current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                    self.peak_memory_tracker = max(self.peak_memory_tracker, current_memory)
                
                time.sleep(1)  # Monitor every second
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous monitoring"""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=2)

class SystemResourceMonitor:
    """Advanced system resource monitoring and alerting"""
    
    def __init__(self):
        self.baseline_metrics: Optional[SystemResourceInfo] = None
        self.alert_thresholds = {
            'memory_usage_percent': 85.0,
            'cpu_usage_percent': 90.0,
            'disk_usage_percent': 90.0
        }
    
    def capture_baseline(self) -> SystemResourceInfo:
        """Capture baseline system metrics"""
        self.baseline_metrics = SystemResourceInfo.capture_current()
        logger.info(f"📊 Baseline captured: {self.baseline_metrics.memory_available_gb:.1f}GB available, "
                   f"{self.baseline_metrics.cpu_count_physical} cores")
        return self.baseline_metrics
    
    def check_resource_health(self) -> Dict[str, Any]:
        """Check current resource health and return status"""
        current = SystemResourceInfo.capture_current()
        alerts = []
        
        # Check thresholds
        if current.memory_usage_percent > self.alert_thresholds['memory_usage_percent']:
            alerts.append(f"High memory usage: {current.memory_usage_percent:.1f}%")
        
        if current.cpu_usage_percent > self.alert_thresholds['cpu_usage_percent']:
            alerts.append(f"High CPU usage: {current.cpu_usage_percent:.1f}%")
        
        if current.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
            alerts.append(f"High disk usage: {current.disk_usage_percent:.1f}%")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"⚠️ {alert}")
        
        return {
            "status": "healthy" if not alerts else "warning",
            "alerts": alerts,
            "current_metrics": current,
            "baseline_metrics": self.baseline_metrics
        }
    
    @staticmethod
    def log_system_info():
        """Log comprehensive system information"""
        info = SystemResourceInfo.capture_current()
        
        logger.info(f"💻 System Info:")
        logger.info(f"   CPU: {info.cpu_count_physical} physical cores, {info.cpu_count_logical} logical cores")
        logger.info(f"   Memory: {info.memory_available_gb:.1f}GB available / {info.memory_total_gb:.1f}GB total ({info.memory_usage_percent:.1f}% used)")
        logger.info(f"   Disk: {info.disk_usage_percent:.1f}% used")
        
        if info.load_average:
            logger.info(f"   Load Average: {info.load_average}")
        
        # Performance recommendations
        if info.memory_usage_percent > 80:
            logger.warning(f"⚠️ High memory usage detected: {info.memory_usage_percent:.1f}%")
            logger.info("💡 Consider enabling memory-efficient processing")
        
        if info.cpu_usage_percent > 80:
            logger.warning(f"⚠️ High CPU usage detected: {info.cpu_usage_percent:.1f}%")
            logger.info("💡 Consider reducing parallel processing cores")

# Decorator classes for different optimization strategies
class PerformanceOptimizer:
    """Advanced performance optimization decorators and utilities"""
    
    @staticmethod
    def auto_optimize(func: Callable) -> Callable:
        """Decorator that automatically optimizes function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = AdvancedPerformanceMonitor()
            monitor.start(func.__name__)
            
            # Pre-execution optimizations
            gc.collect()  # Clean memory before execution
            
            try:
                # Check if function has optimization hints
                if hasattr(func, '_chunk_size_hint'):
                    chunk_size = func._chunk_size_hint
                    if 'chunk_size' not in kwargs:
                        kwargs['chunk_size'] = chunk_size
                
                result = func(*args, **kwargs)
                
                # Post-execution analysis
                metrics = monitor.end()
                if metrics and metrics.duration_seconds > 10:
                    logger.info(f"💡 {func.__name__} could benefit from optimization")
                
                return result
                
            except Exception as e:
                monitor.end()
                logger.error(f"❌ {func.__name__} failed: {e}")
                raise
        
        return wrapper
    
    @staticmethod
    def memory_efficient(threshold_mb: float = 500.0):
        """Decorator for memory-efficient operations"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                monitor = AdvancedPerformanceMonitor()
                monitor.start(f"{func.__name__} (memory-efficient)")
                
                # Enable memory monitoring
                initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                
                try:
                    # Force garbage collection
                    gc.collect()
                    
                    result = func(*args, **kwargs)
                    
                    # Check memory usage
                    final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                    memory_used = final_memory - initial_memory
                    
                    if memory_used > threshold_mb:
                        logger.warning(f"🧠 {func.__name__} used {memory_used:.1f}MB (threshold: {threshold_mb}MB)")
                    
                    monitor.end()
                    return result
                    
                except MemoryError:
                    logger.error(f"💥 {func.__name__} ran out of memory")
                    gc.collect()  # Emergency cleanup
                    raise
                except Exception as e:
                    monitor.end()
                    raise
            
            return wrapper
        return decorator
    
    @staticmethod
    def with_profiling(enable_line_profiling: bool = False):
        """Decorator that adds detailed profiling"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if enable_line_profiling:
                    try:
                        import cProfile
                        import pstats
                        import io
                        
                        profiler = cProfile.Profile()
                        profiler.enable()
                        
                        result = func(*args, **kwargs)
                        
                        profiler.disable()
                        
                        # Generate profile report
                        s = io.StringIO()
                        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                        ps.print_stats(10)  # Top 10 functions
                        
                        logger.info(f"📊 Profile for {func.__name__}:")
                        logger.info(s.getvalue())
                        
                        return result
                        
                    except ImportError:
                        logger.warning("cProfile not available, falling back to basic monitoring")
                        return PerformanceOptimizer.auto_optimize(func)(*args, **kwargs)
                else:
                    return PerformanceOptimizer.auto_optimize(func)(*args, **kwargs)
            
            return wrapper
        return decorator

# Context managers for resource management
@contextmanager
def performance_context(operation_name: str, enable_gc: bool = True):
    """Context manager for performance monitoring"""
    monitor = AdvancedPerformanceMonitor()
    monitor.start(operation_name)
    
    try:
        if enable_gc:
            gc.collect()
        yield monitor
    finally:
        monitor.end()

@contextmanager
def memory_limit_context(limit_mb: float):
    """Context manager that enforces memory limits"""
    initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    def check_memory():
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        if current_memory - initial_memory > limit_mb:
            raise MemoryError(f"Memory limit exceeded: {current_memory - initial_memory:.1f}MB > {limit_mb}MB")
    
    try:
        yield check_memory
    finally:
        gc.collect()

# Legacy compatibility aliases
PerformanceMonitor = AdvancedPerformanceMonitor
SystemInfo = SystemResourceMonitor

# Convenience decorator aliases
performance_warning = PerformanceOptimizer.auto_optimize
memory_efficient_operation = PerformanceOptimizer.memory_efficient

# Utility functions
def optimize_pandas_memory(df):
    """Optimize pandas DataFrame memory usage"""
    initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() <= 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() <= 65535:
                df[col] = df[col].astype('uint16')
            elif df[col].max() <= 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df[col] = df[col].astype('int32')
    
    # Optimize float columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Convert object columns to category where appropriate
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    reduction = (initial_memory - final_memory) / initial_memory * 100
    
    logger.info(f"📉 Memory optimization: {initial_memory:.1f}MB → {final_memory:.1f}MB ({reduction:.1f}% reduction)")
    
    return df

def get_optimal_chunk_size(total_records: int, available_memory_gb: float) -> int:
    """Calculate optimal chunk size based on available memory"""
    # Rough estimate: 1 million records ≈ 100MB
    memory_per_million = 0.1  # GB
    max_records_in_memory = int((available_memory_gb * 0.5) / memory_per_million * 1_000_000)
    
    optimal_chunk = min(total_records, max_records_in_memory, 100_000)
    
    logger.info(f"📊 Optimal chunk size: {optimal_chunk:,} records (Available memory: {available_memory_gb:.1f}GB)")
    
    return optimal_chunk

def force_garbage_collection():
    """Force comprehensive garbage collection"""
    logger.debug("🗑️ Forcing garbage collection...")
    collected = gc.collect()
    logger.debug(f"🗑️ Collected {collected} objects")
    return collected