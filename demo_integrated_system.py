#!/usr/bin/env python3
"""
Liquid AI Vision Kit - Integrated System Demo
Demonstrates full integration of Quantum Task Planning with Liquid Neural Networks
"""

import os
import sys
import time
import asyncio
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_task_planner.core.quantum_engine import QuantumTaskEngine, QuantumTask, TaskPriority
from quantum_task_planner.core.lnn_integration import LNNScheduler
from quantum_task_planner.integration_bridge import LNNQuantumBridge, LNNBridgeConfig


class LiquidVisionDemo:
    """Comprehensive demonstration of the integrated system"""
    
    def __init__(self):
        self.bridge = None
        self.demo_results = []
    
    async def run_full_demo(self):
        """Run comprehensive system demonstration"""
        print("🚀 LIQUID AI VISION KIT - INTEGRATED SYSTEM DEMO")
        print("=" * 80)
        
        # Demo 1: Basic Quantum Task Scheduling
        await self.demo_quantum_scheduling()
        
        # Demo 2: LNN-Guided Adaptive Scheduling  
        await self.demo_lnn_adaptation()
        
        # Demo 3: Vision Task Processing
        await self.demo_vision_processing()
        
        # Demo 4: Real-time System Integration
        await self.demo_realtime_integration()
        
        # Demo 5: Performance Analysis
        self.analyze_performance()
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
    
    async def demo_quantum_scheduling(self):
        """Demonstrate basic quantum task scheduling"""
        print("\n📋 DEMO 1: Quantum Task Scheduling")
        print("-" * 50)
        
        # Create quantum engine
        engine = QuantumTaskEngine(max_workers=2, lnn_integration=False)
        
        # Add resource pools
        engine.add_resource_pool("cpu", capacity=4.0, efficiency=1.2)
        engine.add_resource_pool("memory", capacity=8192, efficiency=1.0)
        
        # Start engine
        engine.start_scheduler()
        
        try:
            # Submit various tasks
            tasks = []
            
            # Critical system tasks
            for i in range(3):
                task = QuantumTask(
                    id=f"critical_task_{i:03d}",
                    name=f"Critical System Task {i}",
                    priority=TaskPriority.CRITICAL,
                    resources={"cpu": 1.0, "memory": 512},
                    estimated_duration=0.5,
                    callback=self.demo_task_callback
                )
                task_id = engine.submit_task(task)
                tasks.append(task_id)
                print(f"✓ Submitted critical task: {task_id}")
            
            # High priority processing tasks
            for i in range(5):
                task = QuantumTask(
                    id=f"processing_task_{i:03d}",
                    name=f"Data Processing Task {i}",
                    priority=TaskPriority.HIGH,
                    resources={"cpu": 0.5, "memory": 256},
                    estimated_duration=1.0,
                    callback=self.demo_task_callback
                )
                task_id = engine.submit_task(task)
                tasks.append(task_id)
            
            print(f"✓ Submitted {len(tasks)} tasks to quantum scheduler")
            
            # Monitor execution
            completed = 0
            timeout = 30
            start_time = time.time()
            
            while completed < len(tasks) and (time.time() - start_time) < timeout:
                status = engine.get_status()
                new_completed = status['completed_tasks']
                
                if new_completed > completed:
                    completed = new_completed
                    print(f"📊 Progress: {completed}/{len(tasks)} tasks completed")
                
                await asyncio.sleep(0.5)
            
            # Final status
            final_status = engine.get_status()
            print(f"🎯 Final Results:")
            print(f"   Completed: {final_status['completed_tasks']}")
            print(f"   Failed: {final_status['failed_tasks']}")
            print(f"   Average wait time: {final_status['metrics']['average_wait_time']:.2f}s")
            
            self.demo_results.append({
                'demo': 'quantum_scheduling',
                'tasks_completed': final_status['completed_tasks'],
                'tasks_failed': final_status['failed_tasks'],
                'execution_time': time.time() - start_time
            })
        
        finally:
            engine.stop_scheduler()
    
    async def demo_lnn_adaptation(self):
        """Demonstrate LNN adaptive scheduling"""
        print("\n🧠 DEMO 2: LNN Adaptive Scheduling")
        print("-" * 50)
        
        # Create LNN scheduler
        lnn_scheduler = LNNScheduler(
            input_features=8,
            hidden_neurons=16,
            output_neurons=4
        )
        
        print("✓ LNN scheduler initialized")
        
        # Simulate multiple scheduling scenarios
        scenarios = [
            {"cpu_load": 0.2, "memory_load": 0.3, "task_complexity": 0.4},
            {"cpu_load": 0.8, "memory_load": 0.6, "task_complexity": 0.7},
            {"cpu_load": 0.5, "memory_load": 0.9, "task_complexity": 0.3},
            {"cpu_load": 0.3, "memory_load": 0.2, "task_complexity": 0.8},
        ]
        
        predictions = []
        
        for i, scenario in enumerate(scenarios):
            # Create task features
            features = np.array([
                0.5,  # quantum_weight
                0.1,  # age
                0.0,  # runtime
                2.0,  # dependencies
                1.0,  # priority
                0.8,  # amplitude
                scenario["cpu_load"],
                scenario["memory_load"]
            ])
            
            # Get LNN prediction
            params = lnn_scheduler.predict(features)
            predictions.append(params)
            
            print(f"🔄 Scenario {i+1}: CPU={scenario['cpu_load']:.1f}, MEM={scenario['memory_load']:.1f}")
            print(f"   LNN Recommendations:")
            print(f"     Execution speed: {params[0]:.2f}")
            print(f"     Resource multiplier: {params[1]:.2f}")
            print(f"     Error tolerance: {params[2]:.2f}")
            print(f"     Retry strategy: {params[3]:.2f}")
            
            # Simulate task execution and feedback
            execution_result = {
                'execution_time': np.random.uniform(0.5, 2.0),
                'resource_usage': params[1] * np.random.uniform(0.8, 1.2),
                'success': True,
                'quality_score': np.random.uniform(0.7, 1.0)
            }
            
            target_metrics = {'target_time': 1.0, 'resource_limit': 1.0}
            
            # Calculate performance score and train
            from quantum_task_planner.core.lnn_integration import calculate_performance_score
            score = calculate_performance_score(None, execution_result, target_metrics)
            lnn_scheduler.train_online(features, execution_result, score)
            
            print(f"   Performance score: {score:.3f}")
            await asyncio.sleep(0.1)
        
        # Show adaptation statistics
        stats = lnn_scheduler.get_adaptation_stats()
        print(f"\n📈 LNN Adaptation Results:")
        print(f"   Training examples: {stats['total_training_examples']}")
        print(f"   Recent performance: {stats['recent_performance']:.3f}")
        print(f"   Performance trend: {stats['performance_trend']:.3f}")
        
        self.demo_results.append({
            'demo': 'lnn_adaptation',
            'training_examples': stats['total_training_examples'],
            'performance': stats['recent_performance'],
            'predictions_made': len(predictions)
        })
    
    async def demo_vision_processing(self):
        """Demonstrate vision task processing"""
        print("\n👁️ DEMO 3: Vision Task Processing")
        print("-" * 50)
        
        # Create bridge with C++ integration disabled for demo
        bridge_config = LNNBridgeConfig(
            enable_cpp_integration=False,  # Disable for demo
            enable_real_time_adaptation=True
        )
        
        self.bridge = LNNQuantumBridge(bridge_config)
        
        if not self.bridge.start_bridge():
            print("❌ Failed to start integration bridge")
            return
        
        try:
            print("✓ LNN-Quantum integration bridge started")
            
            # Submit vision processing tasks
            vision_tasks = []
            for i in range(8):
                task_id = self.bridge.submit_vision_task(
                    f"demo_frame_{i:03d}.jpg",
                    self.vision_task_callback
                )
                vision_tasks.append(task_id)
                print(f"📸 Submitted vision task: {task_id}")
                await asyncio.sleep(0.1)
            
            # Monitor vision processing
            start_time = time.time()
            last_completed = 0
            
            for _ in range(20):  # 10 seconds max
                status = self.bridge.get_bridge_status()
                completed = status['quantum_engine']['completed_tasks']
                
                if completed > last_completed:
                    print(f"🎬 Vision tasks completed: {completed}")
                    last_completed = completed
                
                if completed >= len(vision_tasks):
                    break
                
                await asyncio.sleep(0.5)
            
            # Final vision processing results
            final_status = self.bridge.get_bridge_status()
            print(f"\n🎯 Vision Processing Results:")
            print(f"   Tasks completed: {final_status['quantum_engine']['completed_tasks']}")
            print(f"   System CPU: {final_status['system']['cpu_utilization']*100:.1f}%")
            print(f"   System Memory: {final_status['system']['memory_utilization']*100:.1f}%")
            print(f"   LNN Training Examples: {final_status['lnn_scheduler']['total_training_examples']}")
            
            self.demo_results.append({
                'demo': 'vision_processing',
                'tasks_completed': final_status['quantum_engine']['completed_tasks'],
                'lnn_examples': final_status['lnn_scheduler']['total_training_examples'],
                'execution_time': time.time() - start_time
            })
        
        finally:
            if self.bridge:
                self.bridge.stop_bridge()
    
    async def demo_realtime_integration(self):
        """Demonstrate real-time system integration"""
        print("\n⚡ DEMO 4: Real-time System Integration")
        print("-" * 50)
        
        # Restart bridge for real-time demo
        bridge_config = LNNBridgeConfig(
            enable_cpp_integration=False,
            enable_real_time_adaptation=True,
            update_interval_ms=100  # Fast updates
        )
        
        self.bridge = LNNQuantumBridge(bridge_config)
        
        if not self.bridge.start_bridge():
            print("❌ Failed to start integration bridge")
            return
        
        try:
            print("✓ Real-time integration bridge started")
            
            # Simulate real-time workload
            start_time = time.time()
            task_count = 0
            
            # Submit tasks continuously for 5 seconds
            while time.time() - start_time < 5.0:
                # Submit vision task
                task_id = self.bridge.submit_vision_task(f"realtime_frame_{task_count:04d}.jpg")
                task_count += 1
                
                # Get system status
                status = self.bridge.get_bridge_status()
                
                if task_count % 5 == 0:
                    print(f"🔄 Real-time status (t={time.time()-start_time:.1f}s):")
                    print(f"   Tasks submitted: {task_count}")
                    print(f"   Tasks completed: {status['quantum_engine']['completed_tasks']}")
                    print(f"   Queue size: {status['quantum_engine']['queue_size']}")
                    print(f"   System load: {status['system']['cpu_utilization']*100:.0f}% CPU")
                
                await asyncio.sleep(0.2)  # 5Hz submission rate
            
            # Wait for remaining tasks to complete
            print("📋 Waiting for remaining tasks to complete...")
            for _ in range(10):
                status = self.bridge.get_bridge_status()
                if status['quantum_engine']['queue_size'] == 0:
                    break
                await asyncio.sleep(0.5)
            
            # Final real-time results
            final_status = self.bridge.get_bridge_status()
            total_time = time.time() - start_time
            
            print(f"\n🎯 Real-time Integration Results:")
            print(f"   Total execution time: {total_time:.2f}s")
            print(f"   Tasks submitted: {task_count}")
            print(f"   Tasks completed: {final_status['quantum_engine']['completed_tasks']}")
            print(f"   Throughput: {final_status['quantum_engine']['completed_tasks']/total_time:.1f} tasks/sec")
            print(f"   LNN adaptations: {final_status['lnn_scheduler']['total_training_examples']}")
            
            self.demo_results.append({
                'demo': 'realtime_integration',
                'tasks_submitted': task_count,
                'tasks_completed': final_status['quantum_engine']['completed_tasks'],
                'throughput_tps': final_status['quantum_engine']['completed_tasks']/total_time,
                'execution_time': total_time
            })
        
        finally:
            self.bridge.stop_bridge()
    
    def analyze_performance(self):
        """Analyze overall system performance"""
        print("\n📊 DEMO 5: Performance Analysis")
        print("-" * 50)
        
        if not self.demo_results:
            print("❌ No demo results to analyze")
            return
        
        print("🔍 Comprehensive Performance Analysis:")
        
        total_tasks = 0
        total_time = 0
        total_lnn_examples = 0
        
        for result in self.demo_results:
            demo_name = result['demo'].replace('_', ' ').title()
            print(f"\n   {demo_name}:")
            
            if 'tasks_completed' in result:
                print(f"     Tasks completed: {result['tasks_completed']}")
                total_tasks += result['tasks_completed']
            
            if 'execution_time' in result:
                print(f"     Execution time: {result['execution_time']:.2f}s")
                total_time += result['execution_time']
            
            if 'lnn_examples' in result:
                print(f"     LNN examples: {result['lnn_examples']}")
                total_lnn_examples += result['lnn_examples']
            
            if 'throughput_tps' in result:
                print(f"     Throughput: {result['throughput_tps']:.1f} tasks/sec")
        
        print(f"\n🎯 Overall System Performance:")
        print(f"   Total tasks processed: {total_tasks}")
        print(f"   Total execution time: {total_time:.2f}s")
        print(f"   Average throughput: {total_tasks/max(total_time, 1):.2f} tasks/sec")
        print(f"   LNN learning examples: {total_lnn_examples}")
        print(f"   Successful demos: {len(self.demo_results)}/5")
        
        # Performance grade
        if total_tasks >= 20 and total_lnn_examples >= 10:
            grade = "🏆 EXCELLENT"
        elif total_tasks >= 10 and total_lnn_examples >= 5:
            grade = "⭐ GOOD"
        else:
            grade = "✓ BASIC"
        
        print(f"   Performance grade: {grade}")
    
    def demo_task_callback(self, task):
        """Callback for demo tasks"""
        # Simulate some processing work
        processing_time = task.estimated_duration
        time.sleep(processing_time)
        
        return {
            "status": "completed",
            "processing_time": processing_time,
            "result": f"Demo task {task.name} completed successfully",
            "confidence": np.random.uniform(0.8, 1.0)
        }
    
    def vision_task_callback(self, task):
        """Enhanced callback for vision tasks"""
        # Simulate vision processing
        processing_time = np.random.uniform(0.02, 0.08)  # 20-80ms
        time.sleep(processing_time)
        
        # Simulate vision results
        confidence = np.random.uniform(0.7, 0.95)
        forward_velocity = np.random.uniform(0.5, 3.0)
        yaw_rate = np.random.uniform(-0.5, 0.5)
        
        return {
            "status": "completed",
            "confidence": confidence,
            "forward_velocity": forward_velocity,
            "yaw_rate": yaw_rate,
            "processing_time_ms": processing_time * 1000,
            "image_path": task.metadata.get("image_path", "unknown")
        }


async def main():
    """Run the comprehensive integrated system demo"""
    demo = LiquidVisionDemo()
    
    try:
        await demo.run_full_demo()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if demo.bridge:
            demo.bridge.stop_bridge()


if __name__ == "__main__":
    print("Starting Liquid AI Vision Kit Integrated System Demo...")
    asyncio.run(main())