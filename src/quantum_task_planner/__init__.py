#!/usr/bin/env python3
"""
Quantum Task Planner
Advanced task scheduling and resource management using quantum-inspired algorithms
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__description__ = "Quantum-inspired task planning and execution engine with Liquid Neural Network integration"

# Core components
from .core.quantum_engine import (
    QuantumTaskEngine,
    QuantumTask,
    TaskPriority,
    TaskState,
    ResourcePool,
    QuantumPriorityQueue,
    create_sample_tasks
)

from .core.task_scheduler import (
    QuantumResourceAllocator,
    ResourceType,
    ResourceConstraint,
    AllocationStrategy
)

from .core.lnn_integration import (
    LNNScheduler,
    LNNConfig,
    LiquidNeuron,
    extract_task_features,
    calculate_performance_score
)

# Validation and security
from .validation.task_validator import (
    TaskValidator,
    ValidationReport,
    ValidationResult,
    SecurityLevel
)

# Performance optimization
from .performance.cache_manager import (
    QuantumCacheManager,
    quantum_cache,
    CacheLevel,
    EvictionPolicy
)

# Convenience imports
__all__ = [
    # Core engine
    'QuantumTaskEngine',
    'QuantumTask', 
    'TaskPriority',
    'TaskState',
    'ResourcePool',
    'QuantumPriorityQueue',
    'create_sample_tasks',
    
    # Resource management
    'QuantumResourceAllocator',
    'ResourceType',
    'ResourceConstraint', 
    'AllocationStrategy',
    
    # LNN integration
    'LNNScheduler',
    'LNNConfig',
    'LiquidNeuron',
    'extract_task_features',
    'calculate_performance_score',
    
    # Validation
    'TaskValidator',
    'ValidationReport',
    'ValidationResult',
    'SecurityLevel',
    
    # Performance
    'QuantumCacheManager',
    'quantum_cache',
    'CacheLevel',
    'EvictionPolicy',
    
    # Package metadata
    '__version__',
    '__author__',
    '__description__'
]

def get_version_info():
    """Get detailed version information"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'components': {
            'quantum_engine': 'Core quantum task scheduling engine',
            'task_scheduler': 'Advanced resource allocation with quantum optimization',
            'lnn_integration': 'Liquid Neural Network adaptive scheduling',
            'task_validator': 'Comprehensive validation and security framework',
            'cache_manager': 'Multi-level quantum-inspired caching system'
        },
        'features': [
            'Quantum-inspired task prioritization',
            'Liquid Neural Network adaptive scheduling',
            'Multi-dimensional resource optimization', 
            'Comprehensive security validation',
            'Multi-level caching with quantum eviction policies',
            'Concurrent execution with worker pools',
            'Real-time performance monitoring',
            'Automatic retry with exponential backoff',
            'Dependency resolution and DAG execution'
        ]
    }

def create_default_engine(max_workers=4, enable_lnn=True, enable_cache=True):
    """Create a quantum task engine with sensible defaults"""
    # Create engine with LNN integration
    engine = QuantumTaskEngine(max_workers=max_workers, lnn_integration=enable_lnn)
    
    # Add standard resource pools
    engine.add_resource_pool("cpu", capacity=max_workers * 2.0, efficiency=1.2)
    engine.add_resource_pool("memory", capacity=8192, efficiency=1.0)  # 8GB
    engine.add_resource_pool("disk", capacity=1000, efficiency=0.9)    # 1TB 
    engine.add_resource_pool("network", capacity=1000, efficiency=0.8) # 1Gbps
    
    return engine

def create_validator():
    """Create a task validator with security settings"""
    return TaskValidator()

def create_cache_manager():
    """Create a quantum cache manager"""
    return QuantumCacheManager()


# Package initialization
def initialize_quantum_system():
    """Initialize the quantum task planning system"""
    print(f"Initializing Quantum Task Planner v{__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    
    # Perform any necessary initialization
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Quantum Task Planner initialized successfully")
    
    return True

# Auto-initialize on import
_initialized = initialize_quantum_system()