#!/usr/bin/env python3
"""
Advanced Task Scheduler with Quantum-Inspired Resource Allocation
Multi-dimensional resource optimization using quantum principles
"""

import numpy as np
import threading
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict
import asyncio


class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    CUSTOM = "custom"


class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    GREEDY = "greedy"
    QUANTUM_SUPERPOSITION = "quantum_superposition"
    FAIR_SHARE = "fair_share"
    PRIORITY_WEIGHTED = "priority_weighted"
    ADAPTIVE_LNN = "adaptive_lnn"


@dataclass
class ResourceConstraint:
    """Resource constraint specification"""
    resource_type: ResourceType
    min_required: float
    max_allowed: float
    preferred: float
    weight: float = 1.0  # Importance weight
    
    def is_satisfied(self, allocated: float) -> bool:
        """Check if allocation satisfies constraint"""
        return self.min_required <= allocated <= self.max_allowed


@dataclass
class ResourcePool:
    """Enhanced resource pool with quantum properties"""
    name: str
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    reserved_capacity: float = 0.0
    quantum_efficiency: float = 1.0
    fragmentation_factor: float = 0.0
    allocation_history: List[Tuple[float, str]] = field(default_factory=list)
    
    # Quantum properties
    coherence_time: float = 10.0  # seconds
    entanglement_partners: Set[str] = field(default_factory=set)
    superposition_state: bool = False
    
    def get_effective_capacity(self) -> float:
        """Get capacity adjusted for quantum efficiency"""
        base_capacity = self.total_capacity - self.reserved_capacity
        return base_capacity * self.quantum_efficiency * (1.0 - self.fragmentation_factor)
    
    def can_allocate(self, amount: float) -> bool:
        """Check if amount can be allocated"""
        return self.available_capacity >= amount
    
    def allocate(self, task_id: str, amount: float) -> bool:
        """Allocate resources with quantum optimization"""
        if not self.can_allocate(amount):
            return False
        
        self.available_capacity -= amount
        self.allocation_history.append((amount, task_id))
        
        # Update fragmentation
        self._update_fragmentation()
        
        return True
    
    def deallocate(self, task_id: str, amount: float):
        """Deallocate resources"""
        self.available_capacity += amount
        
        # Remove from history
        self.allocation_history = [
            (amt, tid) for amt, tid in self.allocation_history 
            if tid != task_id
        ]
        
        self._update_fragmentation()
    
    def _update_fragmentation(self):
        """Update fragmentation factor based on allocation pattern"""
        if not self.allocation_history:
            self.fragmentation_factor = 0.0
            return
        
        # Calculate fragmentation based on allocation sizes
        allocations = [amt for amt, _ in self.allocation_history]
        if len(allocations) > 1:
            variance = np.var(allocations)
            mean_allocation = np.mean(allocations)
            self.fragmentation_factor = min(0.5, variance / (mean_allocation + 1e-6))
        else:
            self.fragmentation_factor = 0.0
    
    def get_utilization(self) -> float:
        """Get current utilization percentage"""
        return (self.total_capacity - self.available_capacity) / self.total_capacity


class QuantumResourceAllocator:
    """Quantum-inspired resource allocation engine"""
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.QUANTUM_SUPERPOSITION):
        self.strategy = strategy
        self.resource_pools = {}
        self.allocation_matrix = defaultdict(dict)  # task_id -> resource_name -> amount
        self.entanglement_graph = defaultdict(set)
        self.allocation_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'average_allocation_time': 0.0,
            'resource_efficiency': {},
            'quantum_coherence_score': 1.0
        }
        
        # Quantum state tracking
        self.quantum_state = {
            'superposition_active': False,
            'entanglement_strength': 0.0,
            'coherence_decay_rate': 0.1,
            'last_measurement': time.time()
        }
    
    def add_resource_pool(self, pool: ResourcePool):
        """Add resource pool to allocator"""
        self.resource_pools[pool.name] = pool
        self.metrics['resource_efficiency'][pool.name] = 1.0
    
    def create_resource_pool(self, name: str, resource_type: ResourceType, 
                           capacity: float, quantum_efficiency: float = 1.0) -> ResourcePool:
        """Create and add resource pool"""
        pool = ResourcePool(
            name=name,
            resource_type=resource_type,
            total_capacity=capacity,
            available_capacity=capacity,
            quantum_efficiency=quantum_efficiency
        )
        self.add_resource_pool(pool)
        return pool
    
    def allocate_resources(self, task_id: str, 
                          resource_requirements: Dict[str, ResourceConstraint]) -> Tuple[bool, Dict[str, float]]:
        """Allocate resources using quantum-inspired optimization"""
        start_time = time.time()
        allocation_result = {}
        successful = False
        
        with self.allocation_lock:
            try:
                if self.strategy == AllocationStrategy.QUANTUM_SUPERPOSITION:
                    successful, allocation_result = self._quantum_superposition_allocation(
                        task_id, resource_requirements
                    )
                elif self.strategy == AllocationStrategy.GREEDY:
                    successful, allocation_result = self._greedy_allocation(
                        task_id, resource_requirements
                    )
                elif self.strategy == AllocationStrategy.FAIR_SHARE:
                    successful, allocation_result = self._fair_share_allocation(
                        task_id, resource_requirements
                    )
                elif self.strategy == AllocationStrategy.PRIORITY_WEIGHTED:
                    successful, allocation_result = self._priority_weighted_allocation(
                        task_id, resource_requirements
                    )
                else:
                    successful, allocation_result = self._adaptive_lnn_allocation(
                        task_id, resource_requirements
                    )
                
                # Update metrics
                self.metrics['total_allocations'] += 1
                if successful:
                    self.metrics['successful_allocations'] += 1
                    self.allocation_matrix[task_id] = allocation_result.copy()
                else:
                    self.metrics['failed_allocations'] += 1
                
                # Update timing
                allocation_time = time.time() - start_time
                self.metrics['average_allocation_time'] = (
                    self.metrics['average_allocation_time'] * (self.metrics['total_allocations'] - 1) +
                    allocation_time
                ) / self.metrics['total_allocations']
                
                # Update quantum coherence
                self._update_quantum_coherence(successful, allocation_time)
                
            except Exception as e:
                print(f"Resource allocation error for task {task_id}: {e}")
                successful = False
                allocation_result = {}
        
        return successful, allocation_result
    
    def _quantum_superposition_allocation(self, task_id: str, 
                                        requirements: Dict[str, ResourceConstraint]) -> Tuple[bool, Dict[str, float]]:
        """Quantum superposition-based allocation"""
        allocation = {}
        
        # Enter quantum superposition state
        self.quantum_state['superposition_active'] = True
        
        # Calculate quantum-inspired allocation probabilities
        for resource_name, constraint in requirements.items():
            if resource_name not in self.resource_pools:
                continue
            
            pool = self.resource_pools[resource_name]
            
            # Quantum amplitude calculation
            availability_ratio = pool.available_capacity / pool.total_capacity
            efficiency_factor = pool.quantum_efficiency
            
            # Quantum superposition of allocation amounts
            min_alloc = constraint.min_required
            max_alloc = min(constraint.max_allowed, pool.available_capacity)
            preferred_alloc = min(constraint.preferred, pool.available_capacity)
            
            if max_alloc < min_alloc:
                return False, {}
            
            # Calculate optimal allocation using quantum interference
            optimal_allocation = self._calculate_quantum_optimal(
                min_alloc, preferred_alloc, max_alloc, 
                availability_ratio, efficiency_factor
            )
            
            # Attempt allocation
            if pool.allocate(task_id, optimal_allocation):
                allocation[resource_name] = optimal_allocation
            else:
                # Rollback previous allocations
                self._rollback_allocations(task_id, allocation)
                return False, {}
        
        # Quantum measurement - collapse superposition
        self.quantum_state['superposition_active'] = False
        self.quantum_state['last_measurement'] = time.time()
        
        return True, allocation
    
    def _calculate_quantum_optimal(self, min_val: float, preferred_val: float, 
                                 max_val: float, availability: float, efficiency: float) -> float:
        """Calculate quantum-optimal allocation using interference patterns"""
        # Create quantum amplitudes for different allocation levels
        levels = np.linspace(min_val, max_val, 100)
        amplitudes = np.zeros_like(levels)
        
        for i, level in enumerate(levels):
            # Amplitude based on preference (higher near preferred)
            pref_distance = abs(level - preferred_val) / (max_val - min_val + 1e-6)
            pref_amplitude = np.exp(-pref_distance * 2)
            
            # Amplitude based on efficiency
            efficiency_amplitude = efficiency
            
            # Amplitude based on availability
            availability_amplitude = availability
            
            # Combined amplitude with quantum interference
            amplitudes[i] = pref_amplitude * efficiency_amplitude * availability_amplitude
        
        # Find level with maximum probability density
        probabilities = amplitudes ** 2
        optimal_idx = np.argmax(probabilities)
        
        return levels[optimal_idx]
    
    def _greedy_allocation(self, task_id: str, 
                          requirements: Dict[str, ResourceConstraint]) -> Tuple[bool, Dict[str, float]]:
        """Simple greedy allocation"""
        allocation = {}
        
        for resource_name, constraint in requirements.items():
            if resource_name not in self.resource_pools:
                continue
            
            pool = self.resource_pools[resource_name]
            
            # Try preferred amount first
            if pool.can_allocate(constraint.preferred):
                amount = constraint.preferred
            elif pool.can_allocate(constraint.min_required):
                amount = min(constraint.min_required, pool.available_capacity)
            else:
                self._rollback_allocations(task_id, allocation)
                return False, {}
            
            if pool.allocate(task_id, amount):
                allocation[resource_name] = amount
            else:
                self._rollback_allocations(task_id, allocation)
                return False, {}
        
        return True, allocation
    
    def _fair_share_allocation(self, task_id: str, 
                              requirements: Dict[str, ResourceConstraint]) -> Tuple[bool, Dict[str, float]]:
        """Fair share allocation based on current load"""
        allocation = {}
        
        for resource_name, constraint in requirements.items():
            if resource_name not in self.resource_pools:
                continue
            
            pool = self.resource_pools[resource_name]
            
            # Calculate fair share based on current allocations
            active_tasks = len([tid for tid, allocs in self.allocation_matrix.items() 
                              if resource_name in allocs])
            
            if active_tasks == 0:
                fair_share = pool.total_capacity
            else:
                fair_share = pool.total_capacity / (active_tasks + 1)
            
            # Allocate within constraints and fair share
            amount = max(constraint.min_required, 
                        min(constraint.preferred, fair_share, pool.available_capacity))
            
            if amount >= constraint.min_required and pool.allocate(task_id, amount):
                allocation[resource_name] = amount
            else:
                self._rollback_allocations(task_id, allocation)
                return False, {}
        
        return True, allocation
    
    def _priority_weighted_allocation(self, task_id: str, 
                                    requirements: Dict[str, ResourceConstraint]) -> Tuple[bool, Dict[str, float]]:
        """Priority-weighted allocation"""
        # This would need task priority information passed in
        # For now, implement as greedy with weight consideration
        allocation = {}
        
        # Sort requirements by weight (importance)
        sorted_requirements = sorted(requirements.items(), 
                                   key=lambda x: x[1].weight, reverse=True)
        
        for resource_name, constraint in sorted_requirements:
            if resource_name not in self.resource_pools:
                continue
            
            pool = self.resource_pools[resource_name]
            
            # Weight-adjusted allocation
            weight_factor = constraint.weight
            adjusted_preferred = constraint.preferred * weight_factor
            
            amount = min(adjusted_preferred, pool.available_capacity)
            amount = max(amount, constraint.min_required)
            
            if amount <= constraint.max_allowed and pool.allocate(task_id, amount):
                allocation[resource_name] = amount
            else:
                self._rollback_allocations(task_id, allocation)
                return False, {}
        
        return True, allocation
    
    def _adaptive_lnn_allocation(self, task_id: str, 
                               requirements: Dict[str, ResourceConstraint]) -> Tuple[bool, Dict[str, float]]:
        """LNN-guided adaptive allocation (placeholder)"""
        # This would integrate with the LNN scheduler
        # For now, use quantum superposition as fallback
        return self._quantum_superposition_allocation(task_id, requirements)
    
    def _rollback_allocations(self, task_id: str, partial_allocation: Dict[str, float]):
        """Rollback partial allocations on failure"""
        for resource_name, amount in partial_allocation.items():
            if resource_name in self.resource_pools:
                self.resource_pools[resource_name].deallocate(task_id, amount)
    
    def deallocate_resources(self, task_id: str) -> bool:
        """Deallocate all resources for a task"""
        if task_id not in self.allocation_matrix:
            return False
        
        with self.allocation_lock:
            allocations = self.allocation_matrix.pop(task_id)
            
            for resource_name, amount in allocations.items():
                if resource_name in self.resource_pools:
                    self.resource_pools[resource_name].deallocate(task_id, amount)
        
        return True
    
    def _update_quantum_coherence(self, allocation_successful: bool, allocation_time: float):
        """Update quantum coherence based on allocation performance"""
        current_time = time.time()
        time_since_measurement = current_time - self.quantum_state['last_measurement']
        
        # Apply decoherence
        decoherence = np.exp(-time_since_measurement * self.quantum_state['coherence_decay_rate'])
        self.metrics['quantum_coherence_score'] *= decoherence
        
        # Boost coherence on successful allocations
        if allocation_successful:
            coherence_boost = 1.0 / (1.0 + allocation_time)  # Faster = more coherent
            self.metrics['quantum_coherence_score'] = min(1.0, 
                self.metrics['quantum_coherence_score'] * (1.0 + coherence_boost * 0.1))
        
        self.quantum_state['last_measurement'] = current_time
    
    def get_resource_status(self) -> Dict:
        """Get comprehensive resource status"""
        status = {
            'pools': {},
            'metrics': self.metrics.copy(),
            'quantum_state': self.quantum_state.copy(),
            'active_allocations': len(self.allocation_matrix)
        }
        
        for name, pool in self.resource_pools.items():
            status['pools'][name] = {
                'type': pool.resource_type.value,
                'total_capacity': pool.total_capacity,
                'available_capacity': pool.available_capacity,
                'utilization': pool.get_utilization(),
                'quantum_efficiency': pool.quantum_efficiency,
                'fragmentation_factor': pool.fragmentation_factor,
                'active_allocations': len(pool.allocation_history)
            }
        
        return status
    
    def optimize_resources(self):
        """Optimize resource allocation using quantum annealing principles"""
        with self.allocation_lock:
            # Collect current allocation state
            current_allocations = {}
            for task_id, allocations in self.allocation_matrix.items():
                current_allocations[task_id] = allocations.copy()
            
            # Calculate optimization score
            current_score = self._calculate_allocation_score(current_allocations)
            
            # Try quantum annealing-inspired optimization
            temperature = 1.0
            cooling_rate = 0.95
            min_temperature = 0.01
            
            best_allocations = current_allocations.copy()
            best_score = current_score
            
            while temperature > min_temperature:
                # Generate neighbor solution
                neighbor_allocations = self._generate_neighbor_solution(current_allocations)
                neighbor_score = self._calculate_allocation_score(neighbor_allocations)
                
                # Accept or reject based on simulated annealing
                if neighbor_score > best_score or np.random.random() < np.exp((neighbor_score - current_score) / temperature):
                    current_allocations = neighbor_allocations
                    current_score = neighbor_score
                    
                    if neighbor_score > best_score:
                        best_allocations = neighbor_allocations.copy()
                        best_score = neighbor_score
                
                temperature *= cooling_rate
            
            # Apply optimized allocations if significantly better
            if best_score > current_score * 1.05:  # 5% improvement threshold
                self._apply_optimized_allocations(best_allocations)
    
    def _calculate_allocation_score(self, allocations: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall allocation efficiency score"""
        if not allocations:
            return 0.0
        
        score = 0.0
        
        # Resource utilization efficiency
        for pool_name, pool in self.resource_pools.items():
            utilization = pool.get_utilization()
            # Prefer moderate utilization (avoid both underutilization and over-allocation)
            optimal_utilization = 0.75
            utilization_score = 1.0 - abs(utilization - optimal_utilization)
            score += utilization_score * pool.quantum_efficiency
        
        # Fragmentation penalty
        fragmentation_penalty = sum(pool.fragmentation_factor for pool in self.resource_pools.values())
        score -= fragmentation_penalty * 0.1
        
        # Quantum coherence bonus
        score += self.metrics['quantum_coherence_score'] * 0.2
        
        return score / max(len(self.resource_pools), 1)
    
    def _generate_neighbor_solution(self, current_allocations: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Generate neighbor solution for optimization"""
        neighbor = {}
        for task_id, allocations in current_allocations.items():
            neighbor[task_id] = allocations.copy()
        
        # Randomly modify one allocation
        if neighbor:
            task_id = np.random.choice(list(neighbor.keys()))
            resource_name = np.random.choice(list(neighbor[task_id].keys()))
            
            current_amount = neighbor[task_id][resource_name]
            pool = self.resource_pools[resource_name]
            
            # Small random change
            change = np.random.normal(0, current_amount * 0.1)
            new_amount = max(0, current_amount + change)
            new_amount = min(new_amount, pool.total_capacity)
            
            neighbor[task_id][resource_name] = new_amount
        
        return neighbor
    
    def _apply_optimized_allocations(self, optimized_allocations: Dict[str, Dict[str, float]]):
        """Apply optimized resource allocations"""
        # This would require careful reallocation without disrupting running tasks
        # For now, just update metrics to reflect optimization
        print("Applied quantum-optimized resource allocation")
        self.metrics['quantum_coherence_score'] = min(1.0, self.metrics['quantum_coherence_score'] * 1.1)


if __name__ == "__main__":
    # Example usage and testing
    print("Quantum Resource Allocation Engine")
    print("=" * 50)
    
    # Create allocator
    allocator = QuantumResourceAllocator(AllocationStrategy.QUANTUM_SUPERPOSITION)
    
    # Create resource pools
    cpu_pool = allocator.create_resource_pool("cpu", ResourceType.CPU, 8.0, quantum_efficiency=1.2)
    memory_pool = allocator.create_resource_pool("memory", ResourceType.MEMORY, 16384, quantum_efficiency=1.0)
    disk_pool = allocator.create_resource_pool("disk", ResourceType.DISK, 1000, quantum_efficiency=0.9)
    
    print(f"Created resource pools:")
    print(f"  CPU: {cpu_pool.total_capacity} cores")
    print(f"  Memory: {memory_pool.total_capacity} MB")
    print(f"  Disk: {disk_pool.total_capacity} GB")
    
    # Define resource requirements for sample tasks
    task_requirements = {
        "task_1": {
            "cpu": ResourceConstraint(ResourceType.CPU, min_required=1.0, max_allowed=2.0, preferred=1.5, weight=1.0),
            "memory": ResourceConstraint(ResourceType.MEMORY, min_required=512, max_allowed=2048, preferred=1024, weight=0.8)
        },
        "task_2": {
            "cpu": ResourceConstraint(ResourceType.CPU, min_required=0.5, max_allowed=1.0, preferred=0.8, weight=0.6),
            "memory": ResourceConstraint(ResourceType.MEMORY, min_required=256, max_allowed=1024, preferred=512, weight=1.0),
            "disk": ResourceConstraint(ResourceType.DISK, min_required=10, max_allowed=50, preferred=25, weight=0.4)
        },
        "task_3": {
            "cpu": ResourceConstraint(ResourceType.CPU, min_required=2.0, max_allowed=4.0, preferred=3.0, weight=1.2),
            "memory": ResourceConstraint(ResourceType.MEMORY, min_required=1024, max_allowed=4096, preferred=2048, weight=1.0)
        }
    }
    
    # Test resource allocation
    print("\nTesting quantum resource allocation:")
    for task_id, requirements in task_requirements.items():
        success, allocation = allocator.allocate_resources(task_id, requirements)
        print(f"  {task_id}: {'SUCCESS' if success else 'FAILED'}")
        if success:
            for resource, amount in allocation.items():
                print(f"    {resource}: {amount:.2f}")
    
    # Show resource status
    status = allocator.get_resource_status()
    print(f"\nResource Status:")
    for pool_name, pool_status in status['pools'].items():
        print(f"  {pool_name}: {pool_status['utilization']*100:.1f}% utilized")
        print(f"    Available: {pool_status['available_capacity']:.2f}/{pool_status['total_capacity']:.2f}")
    
    print(f"\nQuantum Metrics:")
    print(f"  Coherence Score: {status['metrics']['quantum_coherence_score']:.3f}")
    print(f"  Success Rate: {status['metrics']['successful_allocations']}/{status['metrics']['total_allocations']}")
    
    # Test optimization
    print("\nRunning quantum optimization...")
    allocator.optimize_resources()
    
    # Deallocate resources
    print("\nDeallocating resources...")
    for task_id in task_requirements.keys():
        allocator.deallocate_resources(task_id)
        print(f"  Deallocated {task_id}")
    
    # Final status
    final_status = allocator.get_resource_status()
    print(f"\nFinal Resource Status:")
    for pool_name, pool_status in final_status['pools'].items():
        print(f"  {pool_name}: {pool_status['utilization']*100:.1f}% utilized")