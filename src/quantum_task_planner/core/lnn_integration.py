#!/usr/bin/env python3
"""
Liquid Neural Network Integration for Quantum Task Planning
Adaptive scheduling decisions using continuous-time neural networks
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from scipy.integrate import odeint


@dataclass
class LNNConfig:
    """Configuration for Liquid Neural Network scheduler"""
    input_features: int = 8
    hidden_neurons: int = 16
    output_neurons: int = 4
    tau: float = 1.0  # Time constant
    leak_rate: float = 0.1
    adaptation_rate: float = 0.01
    dt: float = 0.01  # Integration timestep
    max_iterations: int = 50


class LiquidNeuron:
    """Individual liquid neural network neuron"""
    
    def __init__(self, tau: float = 1.0, leak_rate: float = 0.1):
        self.tau = tau
        self.leak_rate = leak_rate
        self.state = 0.0
        self.adaptation = 0.0
        self.threshold = 0.5
        
    def update(self, input_current: float, dt: float) -> float:
        """Update neuron state using liquid dynamics"""
        # dx/dt = (-leak * x + input) / tau - adaptation
        derivative = (-self.leak_rate * self.state + input_current) / self.tau - self.adaptation
        self.state += dt * derivative
        
        # Update adaptation based on activity
        self.adaptation += dt * 0.01 * max(0, self.state - self.threshold)
        
        # Apply activation function (tanh with adaptation)
        output = np.tanh(self.state - self.adaptation)
        return output
    
    def reset(self):
        """Reset neuron state"""
        self.state = 0.0
        self.adaptation = 0.0


class LNNLayer:
    """Layer of liquid neural network neurons"""
    
    def __init__(self, input_size: int, output_size: int, config: LNNConfig):
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        
        # Initialize neurons
        self.neurons = [
            LiquidNeuron(tau=config.tau, leak_rate=config.leak_rate)
            for _ in range(output_size)
        ]
        
        # Initialize weights with Xavier initialization
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(output_size, input_size) * scale
        self.biases = np.zeros(output_size)
        
        # Recurrent connections within layer
        self.recurrent_weights = np.random.randn(output_size, output_size) * 0.1
        np.fill_diagonal(self.recurrent_weights, 0)  # No self-connections
        
        self.last_output = np.zeros(output_size)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through liquid layer"""
        outputs = np.zeros(self.output_size)
        
        for i, neuron in enumerate(self.neurons):
            # Compute input current
            feedforward = np.dot(self.weights[i], inputs) + self.biases[i]
            recurrent = np.dot(self.recurrent_weights[i], self.last_output)
            total_input = feedforward + recurrent
            
            # Update neuron
            outputs[i] = neuron.update(total_input, self.config.dt)
        
        self.last_output = outputs.copy()
        return outputs
    
    def reset(self):
        """Reset all neurons in layer"""
        for neuron in self.neurons:
            neuron.reset()
        self.last_output.fill(0)


class LNNScheduler:
    """Liquid Neural Network for adaptive task scheduling"""
    
    def __init__(self, input_features: int = 8, hidden_neurons: int = 16, 
                 output_neurons: int = 4, config: Optional[LNNConfig] = None):
        self.config = config or LNNConfig(
            input_features=input_features,
            hidden_neurons=hidden_neurons,
            output_neurons=output_neurons
        )
        
        # Network architecture
        self.input_layer = LNNLayer(input_features, hidden_neurons, self.config)
        self.hidden_layer = LNNLayer(hidden_neurons, hidden_neurons, self.config)
        self.output_layer = LNNLayer(hidden_neurons, output_neurons, self.config)
        
        # Training history
        self.training_data = []
        self.performance_history = []
        self.adaptation_rate = self.config.adaptation_rate
        
        # Feature normalization
        self.input_mean = np.zeros(input_features)
        self.input_std = np.ones(input_features)
        
    def normalize_input(self, features: np.ndarray) -> np.ndarray:
        """Normalize input features"""
        return (features - self.input_mean) / (self.input_std + 1e-8)
    
    def predict(self, task_features: np.ndarray) -> np.ndarray:
        """Predict optimal scheduling parameters"""
        # Normalize inputs
        normalized_features = self.normalize_input(task_features)
        
        # Reset network state for clean inference
        self.reset_state()
        
        # Run liquid dynamics for multiple timesteps
        final_output = None
        for _ in range(self.config.max_iterations):
            # Forward pass through layers
            hidden1 = self.input_layer.forward(normalized_features)
            hidden2 = self.hidden_layer.forward(hidden1)
            output = self.output_layer.forward(hidden2)
            final_output = output
        
        # Map outputs to scheduling parameters
        scheduling_params = self._map_output_to_params(final_output)
        return scheduling_params
    
    def _map_output_to_params(self, raw_output: np.ndarray) -> np.ndarray:
        """Map raw network output to meaningful scheduling parameters"""
        # Apply different activation functions for different parameters
        params = np.zeros_like(raw_output)
        
        # Execution speed multiplier (0.5 to 2.0)
        params[0] = 0.5 + 1.5 * (1 + np.tanh(raw_output[0])) / 2
        
        # Resource multiplier (0.8 to 1.5)  
        params[1] = 0.8 + 0.7 * (1 + np.tanh(raw_output[1])) / 2
        
        # Error tolerance (0.1 to 1.0)
        params[2] = 0.1 + 0.9 * (1 + np.tanh(raw_output[2])) / 2
        
        # Retry strategy (0 to 3)
        params[3] = 3 * (1 + np.tanh(raw_output[3])) / 2
        
        return params
    
    def train_online(self, task_features: np.ndarray, task_result: Dict, 
                    performance_score: float):
        """Online learning from task execution results"""
        # Store training example
        self.training_data.append({
            'features': task_features.copy(),
            'result': task_result.copy(),
            'score': performance_score,
            'timestamp': time.time()
        })
        
        # Update input normalization statistics
        self._update_normalization(task_features)
        
        # Adapt network based on performance
        if len(self.training_data) > 1:
            self._adapt_weights(task_features, performance_score)
        
        self.performance_history.append(performance_score)
        
        # Keep only recent training data
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-800:]
    
    def _update_normalization(self, features: np.ndarray):
        """Update running statistics for input normalization"""
        if len(self.training_data) == 1:
            self.input_mean = features.copy()
            self.input_std = np.ones_like(features)
        else:
            # Exponential moving average
            alpha = 0.01
            self.input_mean = (1 - alpha) * self.input_mean + alpha * features
            
            diff = features - self.input_mean
            self.input_std = (1 - alpha) * self.input_std + alpha * np.abs(diff)
    
    def _adapt_weights(self, features: np.ndarray, score: float):
        """Adapt network weights based on performance feedback"""
        if len(self.performance_history) < 2:
            return
        
        # Calculate performance gradient
        recent_scores = self.performance_history[-10:]
        performance_trend = np.mean(recent_scores) - np.mean(recent_scores[:-5] if len(recent_scores) > 5 else recent_scores[:1])
        
        # Adapt based on performance trend
        adaptation_strength = self.adaptation_rate * np.tanh(performance_trend)
        
        # Simple weight adaptation (reward/punishment)
        if score > 0.7:  # Good performance
            self._strengthen_connections(features, adaptation_strength)
        elif score < 0.3:  # Poor performance
            self._weaken_connections(features, adaptation_strength)
    
    def _strengthen_connections(self, features: np.ndarray, strength: float):
        """Strengthen connections that led to good performance"""
        # Increase weights for active connections
        normalized_features = self.normalize_input(features)
        
        # Strengthen input layer connections
        for i in range(self.input_layer.output_size):
            active_inputs = normalized_features > 0.1
            self.input_layer.weights[i][active_inputs] += strength * 0.01
            
        # Add small random exploration
        self.input_layer.weights += np.random.randn(*self.input_layer.weights.shape) * strength * 0.001
    
    def _weaken_connections(self, features: np.ndarray, strength: float):
        """Weaken connections that led to poor performance"""
        normalized_features = self.normalize_input(features)
        
        # Weaken problematic connections
        for i in range(self.input_layer.output_size):
            active_inputs = normalized_features > 0.1
            self.input_layer.weights[i][active_inputs] -= abs(strength) * 0.01
            
        # Ensure weights don't become too small
        self.input_layer.weights = np.clip(self.input_layer.weights, -5.0, 5.0)
    
    def reset_state(self):
        """Reset network state for clean inference"""
        self.input_layer.reset()
        self.hidden_layer.reset()
        self.output_layer.reset()
    
    def get_adaptation_stats(self) -> Dict:
        """Get statistics about network adaptation"""
        if not self.performance_history:
            return {"message": "No training data available"}
        
        recent_performance = np.mean(self.performance_history[-50:]) if len(self.performance_history) >= 50 else np.mean(self.performance_history)
        
        return {
            'total_training_examples': len(self.training_data),
            'recent_performance': recent_performance,
            'performance_trend': np.mean(np.diff(self.performance_history[-20:])) if len(self.performance_history) > 20 else 0,
            'input_mean': self.input_mean.tolist(),
            'input_std': self.input_std.tolist(),
            'adaptation_rate': self.adaptation_rate,
            'network_layers': {
                'input_neurons': self.config.input_features,
                'hidden_neurons': self.config.hidden_neurons, 
                'output_neurons': self.config.output_neurons
            }
        }
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        model_data = {
            'config': {
                'input_features': self.config.input_features,
                'hidden_neurons': self.config.hidden_neurons,
                'output_neurons': self.config.output_neurons,
                'tau': self.config.tau,
                'leak_rate': self.config.leak_rate,
                'adaptation_rate': self.config.adaptation_rate
            },
            'weights': {
                'input_layer': {
                    'weights': self.input_layer.weights.tolist(),
                    'biases': self.input_layer.biases.tolist(),
                    'recurrent_weights': self.input_layer.recurrent_weights.tolist()
                },
                'hidden_layer': {
                    'weights': self.hidden_layer.weights.tolist(),
                    'biases': self.hidden_layer.biases.tolist(),
                    'recurrent_weights': self.hidden_layer.recurrent_weights.tolist()
                },
                'output_layer': {
                    'weights': self.output_layer.weights.tolist(),
                    'biases': self.output_layer.biases.tolist(),
                    'recurrent_weights': self.output_layer.recurrent_weights.tolist()
                }
            },
            'normalization': {
                'input_mean': self.input_mean.tolist(),
                'input_std': self.input_std.tolist()
            },
            'performance_history': self.performance_history[-100:]  # Keep recent history
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Restore configuration
            config_data = model_data['config']
            self.config = LNNConfig(
                input_features=config_data['input_features'],
                hidden_neurons=config_data['hidden_neurons'],
                output_neurons=config_data['output_neurons'],
                tau=config_data['tau'],
                leak_rate=config_data['leak_rate'],
                adaptation_rate=config_data['adaptation_rate']
            )
            
            # Rebuild network with loaded config
            self.input_layer = LNNLayer(self.config.input_features, self.config.hidden_neurons, self.config)
            self.hidden_layer = LNNLayer(self.config.hidden_neurons, self.config.hidden_neurons, self.config)
            self.output_layer = LNNLayer(self.config.hidden_neurons, self.config.output_neurons, self.config)
            
            # Restore weights
            weights_data = model_data['weights']
            
            self.input_layer.weights = np.array(weights_data['input_layer']['weights'])
            self.input_layer.biases = np.array(weights_data['input_layer']['biases'])
            self.input_layer.recurrent_weights = np.array(weights_data['input_layer']['recurrent_weights'])
            
            self.hidden_layer.weights = np.array(weights_data['hidden_layer']['weights'])
            self.hidden_layer.biases = np.array(weights_data['hidden_layer']['biases'])
            self.hidden_layer.recurrent_weights = np.array(weights_data['hidden_layer']['recurrent_weights'])
            
            self.output_layer.weights = np.array(weights_data['output_layer']['weights'])
            self.output_layer.biases = np.array(weights_data['output_layer']['biases'])
            self.output_layer.recurrent_weights = np.array(weights_data['output_layer']['recurrent_weights'])
            
            # Restore normalization
            norm_data = model_data['normalization']
            self.input_mean = np.array(norm_data['input_mean'])
            self.input_std = np.array(norm_data['input_std'])
            
            # Restore performance history
            self.performance_history = model_data.get('performance_history', [])
            
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False


def extract_task_features(task, system_state: Dict) -> np.ndarray:
    """Extract features from task and system state for LNN input"""
    features = np.zeros(8)
    
    # Task-specific features
    features[0] = task.quantum_weight / 1000.0  # Normalized priority weight
    features[1] = min(task.age / 3600.0, 1.0)   # Age in hours (capped at 1)
    features[2] = min(task.runtime / 300.0, 1.0)  # Runtime in minutes (capped at 5)
    features[3] = len(task.dependencies) / 10.0  # Number of dependencies
    features[4] = task.priority.value / 3.0      # Priority level
    features[5] = task.amplitude                 # Quantum amplitude
    
    # System state features
    features[6] = system_state.get('cpu_utilization', 0.5)  # CPU utilization
    features[7] = system_state.get('memory_utilization', 0.5)  # Memory utilization
    
    return features


def calculate_performance_score(task, execution_result: Dict, target_metrics: Dict) -> float:
    """Calculate performance score for LNN training"""
    score = 0.0
    weight_sum = 0.0
    
    # Execution time performance
    if 'execution_time' in execution_result and 'target_time' in target_metrics:
        time_ratio = target_metrics['target_time'] / max(execution_result['execution_time'], 0.1)
        time_score = min(1.0, time_ratio)  # Better if faster than target
        score += time_score * 0.3
        weight_sum += 0.3
    
    # Resource efficiency
    if 'resource_usage' in execution_result and 'resource_limit' in target_metrics:
        usage_ratio = execution_result['resource_usage'] / target_metrics['resource_limit']
        resource_score = max(0.0, 1.0 - usage_ratio)  # Better if uses less resources
        score += resource_score * 0.2
        weight_sum += 0.2
    
    # Success rate
    if execution_result.get('success', True):
        score += 1.0 * 0.3
        weight_sum += 0.3
    
    # Quality metrics
    if 'quality_score' in execution_result:
        score += execution_result['quality_score'] * 0.2
        weight_sum += 0.2
    
    # Normalize score
    return score / max(weight_sum, 1.0) if weight_sum > 0 else 0.5


if __name__ == "__main__":
    # Example usage and testing
    print("LNN Integration for Quantum Task Planning")
    print("=" * 50)
    
    # Create LNN scheduler
    scheduler = LNNScheduler(
        input_features=8,
        hidden_neurons=16,
        output_neurons=4
    )
    
    # Simulate task features
    sample_features = np.array([
        0.5,   # quantum_weight (normalized)
        0.1,   # age (hours)
        0.0,   # runtime (minutes)
        2.0,   # dependencies  
        1.0,   # priority
        0.8,   # amplitude
        0.6,   # cpu_utilization
        0.4    # memory_utilization
    ])
    
    # Test prediction
    scheduling_params = scheduler.predict(sample_features)
    print("Scheduling Parameters:")
    print(f"  Execution speed: {scheduling_params[0]:.2f}")
    print(f"  Resource multiplier: {scheduling_params[1]:.2f}")
    print(f"  Error tolerance: {scheduling_params[2]:.2f}")
    print(f"  Retry strategy: {scheduling_params[3]:.2f}")
    
    # Simulate online learning
    for i in range(10):
        # Simulate task execution result
        execution_result = {
            'execution_time': np.random.uniform(1.0, 5.0),
            'resource_usage': np.random.uniform(0.3, 1.2),
            'success': np.random.choice([True, False], p=[0.8, 0.2]),
            'quality_score': np.random.uniform(0.6, 1.0)
        }
        
        target_metrics = {
            'target_time': 3.0,
            'resource_limit': 1.0
        }
        
        # Calculate performance score
        score = calculate_performance_score(None, execution_result, target_metrics)
        
        # Train online
        scheduler.train_online(sample_features, execution_result, score)
        
        print(f"Training iteration {i+1}: Performance score = {score:.3f}")
    
    # Get adaptation statistics
    stats = scheduler.get_adaptation_stats()
    print(f"\nAdaptation Statistics:")
    print(f"  Training examples: {stats['total_training_examples']}")
    print(f"  Recent performance: {stats['recent_performance']:.3f}")
    print(f"  Performance trend: {stats['performance_trend']:.3f}")
    
    # Test model save/load
    scheduler.save_model("test_lnn_model.json")
    print("\nModel saved to test_lnn_model.json")
    
    # Create new scheduler and load model
    new_scheduler = LNNScheduler()
    if new_scheduler.load_model("test_lnn_model.json"):
        print("Model loaded successfully")
        
        # Test loaded model
        new_params = new_scheduler.predict(sample_features)
        print("Loaded model predictions:")
        print(f"  Execution speed: {new_params[0]:.2f}")
        print(f"  Resource multiplier: {new_params[1]:.2f}")
        print(f"  Error tolerance: {new_params[2]:.2f}")
        print(f"  Retry strategy: {new_params[3]:.2f}")
    else:
        print("Failed to load model")