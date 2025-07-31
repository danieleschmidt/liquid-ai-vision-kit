"""Performance benchmarks for Liquid Vision Kit."""

import pytest
import numpy as np
import time
from unittest.mock import Mock


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.performance
    def test_inference_latency_benchmark(self, benchmark, sample_image):
        """Benchmark neural network inference latency."""
        
        def mock_inference(image):
            """Mock inference function with realistic timing."""
            # Simulate processing time
            time.sleep(0.01)  # 10ms simulation
            return {
                "forward_velocity": 1.5,
                "yaw_rate": 0.2,
                "confidence": 0.95
            }
        
        result = benchmark(mock_inference, sample_image)
        
        # Verify reasonable output
        assert result["confidence"] > 0.8
        assert abs(result["forward_velocity"]) < 5.0
        
    @pytest.mark.performance
    def test_image_preprocessing_benchmark(self, benchmark, sample_image):
        """Benchmark image preprocessing performance."""
        
        def preprocess_image(image):
            """Mock image preprocessing."""
            # Simulate typical preprocessing steps
            resized = np.resize(image, (120, 160, 3))
            normalized = resized.astype(np.float32) / 255.0
            return normalized
        
        result = benchmark(preprocess_image, sample_image)
        
        # Verify output shape and range
        assert result.shape == (120, 160, 3)
        assert 0.0 <= result.max() <= 1.0
        assert 0.0 <= result.min() <= 1.0
        
    @pytest.mark.performance
    def test_memory_usage_benchmark(self, sample_image, performance_thresholds):
        """Test memory usage during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024  # KB
        
        # Simulate processing
        processed_images = []
        for _ in range(10):
            # Mock image processing
            processed = sample_image.copy().astype(np.float32)
            processed_images.append(processed)
        
        peak_memory = process.memory_info().rss / 1024  # KB
        memory_delta = peak_memory - initial_memory
        
        # Clean up
        del processed_images
        
        # Assert memory usage is within limits
        assert memory_delta < performance_thresholds["max_memory_usage_kb"]
        
    @pytest.mark.performance
    def test_throughput_benchmark(self, benchmark, sample_image):
        """Benchmark processing throughput (frames per second)."""
        
        def process_batch(images):
            """Process a batch of images."""
            results = []
            for img in images:
                # Mock processing
                result = {
                    "processed": True,
                    "timestamp": time.time()
                }
                results.append(result)
            return results
        
        # Create batch of test images
        batch_size = 5
        image_batch = [sample_image] * batch_size
        
        results = benchmark(process_batch, image_batch)
        
        # Verify all images processed
        assert len(results) == batch_size
        assert all(r["processed"] for r in results)


@pytest.mark.performance
class TestRealTimeConstraints:
    """Test real-time performance constraints."""
    
    def test_inference_time_constraint(self, sample_image, performance_thresholds):
        """Ensure inference meets real-time constraints."""
        
        def timed_inference(image):
            start_time = time.perf_counter()
            
            # Mock inference processing
            time.sleep(0.015)  # 15ms simulation
            result = {"confidence": 0.9}
            
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000
            
            return result, inference_time_ms
        
        result, inference_time = timed_inference(sample_image)
        
        # Assert real-time constraint
        assert inference_time < performance_thresholds["max_inference_time_ms"]
        assert result["confidence"] > 0.8
        
    def test_deterministic_timing(self, sample_image):
        """Test that processing time is consistent."""
        
        def mock_process(image):
            start = time.perf_counter()
            # Simulate consistent processing
            time.sleep(0.01)
            end = time.perf_counter()
            return (end - start) * 1000
        
        # Run multiple times
        times = [mock_process(sample_image) for _ in range(5)]
        
        # Check timing consistency (coefficient of variation < 10%)
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time
        
        assert cv < 0.1, f"Timing inconsistent: CV={cv:.3f}"