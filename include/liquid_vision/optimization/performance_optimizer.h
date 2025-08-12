#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <unordered_map>
#include <chrono>

namespace LiquidVision {

/**
 * Thread-safe object pool for memory optimization
 */
template<typename T>
class ObjectPool {
private:
    std::queue<std::unique_ptr<T>> pool_;
    std::mutex mutex_;
    std::function<std::unique_ptr<T>()> factory_;
    size_t max_size_;
    std::atomic<size_t> total_created_{0};

public:
    explicit ObjectPool(std::function<std::unique_ptr<T>()> factory, size_t max_size = 100)
        : factory_(factory), max_size_(max_size) {}

    class PooledObject {
    private:
        std::unique_ptr<T> object_;
        ObjectPool<T>* pool_;

    public:
        PooledObject(std::unique_ptr<T> obj, ObjectPool<T>* pool) 
            : object_(std::move(obj)), pool_(pool) {}
        
        ~PooledObject() {
            if (pool_ && object_) {
                pool_->return_object(std::move(object_));
            }
        }
        
        T* get() { return object_.get(); }
        T& operator*() { return *object_; }
        T* operator->() { return object_.get(); }
        
        // Move constructor
        PooledObject(PooledObject&& other) noexcept
            : object_(std::move(other.object_)), pool_(other.pool_) {
            other.pool_ = nullptr;
        }
        
        // Delete copy constructor and assignment
        PooledObject(const PooledObject&) = delete;
        PooledObject& operator=(const PooledObject&) = delete;
        PooledObject& operator=(PooledObject&&) = delete;
    };

    PooledObject acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (pool_.empty()) {
            total_created_++;
            return PooledObject(factory_(), this);
        }
        
        auto obj = std::move(pool_.front());
        pool_.pop();
        return PooledObject(std::move(obj), this);
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }
    
    size_t total_created() const {
        return total_created_.load();
    }

private:
    void return_object(std::unique_ptr<T> obj) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (pool_.size() < max_size_) {
            pool_.push(std::move(obj));
        }
        // Otherwise, let it be destroyed
    }
};

/**
 * High-performance cache with LRU eviction
 */
template<typename Key, typename Value>
class LRUCache {
private:
    struct CacheItem {
        Value value;
        typename std::list<Key>::iterator list_iter;
        std::chrono::steady_clock::time_point access_time;
    };
    
    size_t capacity_;
    std::unordered_map<Key, CacheItem> cache_;
    std::list<Key> access_order_;
    mutable std::shared_mutex mutex_;
    
    // Statistics
    std::atomic<size_t> hits_{0};
    std::atomic<size_t> misses_{0};

public:
    explicit LRUCache(size_t capacity) : capacity_(capacity) {}
    
    void put(const Key& key, const Value& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Update existing item
            it->second.value = value;
            it->second.access_time = std::chrono::steady_clock::now();
            
            // Move to front of access order
            access_order_.erase(it->second.list_iter);
            access_order_.push_front(key);
            it->second.list_iter = access_order_.begin();
        } else {
            // Add new item
            if (cache_.size() >= capacity_) {
                // Evict least recently used
                const Key& lru_key = access_order_.back();
                cache_.erase(lru_key);
                access_order_.pop_back();
            }
            
            access_order_.push_front(key);
            cache_[key] = {value, access_order_.begin(), std::chrono::steady_clock::now()};
        }
    }
    
    std::optional<Value> get(const Key& key) {
        std::shared_lock<std::shared_mutex> read_lock(mutex_);
        
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            misses_++;
            return std::nullopt;
        }
        
        hits_++;
        
        // Update access order (need to upgrade to unique lock)
        read_lock.unlock();
        std::unique_lock<std::shared_mutex> write_lock(mutex_);
        
        // Check again after acquiring write lock
        it = cache_.find(key);
        if (it == cache_.end()) {
            misses_++;
            return std::nullopt;
        }
        
        it->second.access_time = std::chrono::steady_clock::now();
        
        // Move to front
        access_order_.erase(it->second.list_iter);
        access_order_.push_front(key);
        it->second.list_iter = access_order_.begin();
        
        return it->second.value;
    }
    
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        cache_.clear();
        access_order_.clear();
    }
    
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_.size();
    }
    
    double hit_rate() const {
        size_t h = hits_.load();
        size_t m = misses_.load();
        return (h + m) > 0 ? static_cast<double>(h) / (h + m) : 0.0;
    }
    
    void reset_stats() {
        hits_.store(0);
        misses_.store(0);
    }
};

/**
 * Thread pool for concurrent processing
 */
class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    
    // Statistics
    std::atomic<size_t> tasks_processed_{0};
    std::atomic<size_t> tasks_queued_{0};

public:
    explicit ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] { 
                            return stop_.load() || !tasks_.empty(); 
                        });
                        
                        if (stop_.load() && tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    
                    task();
                    tasks_processed_++;
                }
            });
        }
    }
    
    ~ThreadPool() {
        stop_.store(true);
        condition_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            if (stop_.load()) {
                throw std::runtime_error("Cannot enqueue task on stopped ThreadPool");
            }
            
            tasks_.emplace([task] { (*task)(); });
            tasks_queued_++;
        }
        
        condition_.notify_one();
        return result;
    }
    
    size_t queue_size() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }
    
    size_t threads_count() const {
        return workers_.size();
    }
    
    size_t tasks_processed() const {
        return tasks_processed_.load();
    }
    
    size_t tasks_queued() const {
        return tasks_queued_.load();
    }
};

/**
 * Memory pool for efficient allocation
 */
class MemoryPool {
private:
    struct Block {
        Block* next;
    };
    
    std::vector<std::unique_ptr<uint8_t[]>> chunks_;
    Block* free_list_;
    std::mutex mutex_;
    
    size_t block_size_;
    size_t blocks_per_chunk_;
    size_t chunk_size_;
    
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> blocks_in_use_{0};

public:
    MemoryPool(size_t block_size, size_t blocks_per_chunk = 1024)
        : block_size_(align_size(block_size)), 
          blocks_per_chunk_(blocks_per_chunk),
          chunk_size_(block_size_ * blocks_per_chunk_),
          free_list_(nullptr) {
        
        allocate_chunk();
    }
    
    ~MemoryPool() = default;
    
    void* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!free_list_) {
            allocate_chunk();
        }
        
        if (!free_list_) {
            throw std::bad_alloc();
        }
        
        Block* block = free_list_;
        free_list_ = free_list_->next;
        
        blocks_in_use_++;
        return block;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        Block* block = static_cast<Block*>(ptr);
        block->next = free_list_;
        free_list_ = block;
        
        blocks_in_use_--;
    }
    
    size_t block_size() const { return block_size_; }
    size_t blocks_in_use() const { return blocks_in_use_.load(); }
    size_t total_allocated() const { return total_allocated_.load(); }

private:
    size_t align_size(size_t size) const {
        const size_t alignment = sizeof(void*);
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
    void allocate_chunk() {
        auto chunk = std::make_unique<uint8_t[]>(chunk_size_);
        uint8_t* chunk_ptr = chunk.get();
        
        // Initialize free list for this chunk
        for (size_t i = 0; i < blocks_per_chunk_; ++i) {
            Block* block = reinterpret_cast<Block*>(chunk_ptr + i * block_size_);
            block->next = free_list_;
            free_list_ = block;
        }
        
        chunks_.push_back(std::move(chunk));
        total_allocated_ += blocks_per_chunk_;
    }
};

/**
 * Performance-optimized frame processor
 */
class OptimizedFrameProcessor {
private:
    std::unique_ptr<ThreadPool> thread_pool_;
    LRUCache<std::string, std::vector<float>> preprocessing_cache_;
    ObjectPool<std::vector<float>> buffer_pool_;
    MemoryPool memory_pool_;
    
    struct ProcessingStats {
        std::atomic<size_t> frames_processed{0};
        std::atomic<double> total_processing_time{0.0};
        std::atomic<size_t> cache_hits{0};
        std::atomic<size_t> parallel_tasks{0};
    } stats_;

public:
    OptimizedFrameProcessor(size_t num_threads = std::thread::hardware_concurrency())
        : thread_pool_(std::make_unique<ThreadPool>(num_threads)),
          preprocessing_cache_(1000),  // Cache last 1000 processed frames
          buffer_pool_([]() { return std::make_unique<std::vector<float>>(); }, 50),
          memory_pool_(1024 * 1024, 16) {  // 1MB blocks, 16 blocks per chunk
    }
    
    // Async frame processing with caching and pooling
    std::future<std::vector<float>> process_frame_async(
        const std::vector<uint8_t>& frame_data,
        int width, int height, int channels) {
        
        // Generate cache key based on frame characteristics
        std::string cache_key = generate_cache_key(frame_data, width, height, channels);
        
        // Check cache first
        auto cached_result = preprocessing_cache_.get(cache_key);
        if (cached_result.has_value()) {
            stats_.cache_hits++;
            std::promise<std::vector<float>> promise;
            promise.set_value(*cached_result);
            return promise.get_future();
        }
        
        // Submit to thread pool for processing
        return thread_pool_->enqueue([this, frame_data, width, height, channels, cache_key]() {
            return process_frame_internal(frame_data, width, height, channels, cache_key);
        });
    }
    
    // Batch processing for multiple frames
    std::vector<std::future<std::vector<float>>> process_frames_batch(
        const std::vector<std::tuple<std::vector<uint8_t>, int, int, int>>& frames) {
        
        std::vector<std::future<std::vector<float>>> futures;
        futures.reserve(frames.size());
        
        for (const auto& [data, w, h, c] : frames) {
            futures.push_back(process_frame_async(data, w, h, c));
        }
        
        return futures;
    }
    
    struct PerformanceMetrics {
        size_t frames_processed;
        double average_processing_time_ms;
        double cache_hit_rate;
        size_t active_threads;
        size_t queue_size;
        size_t memory_usage_mb;
    };
    
    PerformanceMetrics get_performance_metrics() const {
        double avg_time = stats_.frames_processed.load() > 0 ? 
            stats_.total_processing_time.load() / stats_.frames_processed.load() : 0.0;
        
        return {
            stats_.frames_processed.load(),
            avg_time * 1000.0,  // Convert to ms
            preprocessing_cache_.hit_rate(),
            thread_pool_->threads_count(),
            thread_pool_->queue_size(),
            (memory_pool_.total_allocated() * memory_pool_.block_size()) / (1024 * 1024)
        };
    }

private:
    std::string generate_cache_key(const std::vector<uint8_t>& data, 
                                  int width, int height, int channels) {
        // Simple hash-based key (in production, use more sophisticated hash)
        size_t hash1 = std::hash<size_t>{}(data.size());
        size_t hash2 = std::hash<int>{}(width);
        size_t hash3 = std::hash<int>{}(height);
        size_t hash4 = std::hash<int>{}(channels);
        
        // Combine hashes
        size_t combined = hash1 ^ (hash2 << 1) ^ (hash3 << 2) ^ (hash4 << 3);
        
        // Add simple content hash (sample a few pixels)
        if (!data.empty()) {
            size_t stride = std::max(1UL, data.size() / 16);
            for (size_t i = 0; i < data.size(); i += stride) {
                combined ^= std::hash<uint8_t>{}(data[i]) + 0x9e3779b9 + (combined << 6) + (combined >> 2);
            }
        }
        
        return std::to_string(combined);
    }
    
    std::vector<float> process_frame_internal(const std::vector<uint8_t>& frame_data,
                                            int width, int height, int channels,
                                            const std::string& cache_key) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Get buffer from pool
        auto buffer = buffer_pool_.acquire();
        buffer->clear();
        buffer->reserve(width * height);
        
        // Process frame (resize, normalize, etc.)
        std::vector<float> result;
        result.reserve(160 * 120);  // Target size
        
        // Simple resize and normalize (optimized version)
        const int target_w = 160, target_h = 120;
        const float x_ratio = static_cast<float>(width) / target_w;
        const float y_ratio = static_cast<float>(height) / target_h;
        
        for (int y = 0; y < target_h; ++y) {
            for (int x = 0; x < target_w; ++x) {
                int src_x = static_cast<int>(x * x_ratio);
                int src_y = static_cast<int>(y * y_ratio);
                int src_idx = (src_y * width + src_x) * channels;
                
                if (src_idx < frame_data.size()) {
                    // Convert to grayscale if needed
                    float pixel_value;
                    if (channels == 3) {
                        pixel_value = 0.299f * frame_data[src_idx] + 
                                     0.587f * frame_data[src_idx + 1] + 
                                     0.114f * frame_data[src_idx + 2];
                    } else {
                        pixel_value = frame_data[src_idx];
                    }
                    
                    // Normalize to [-1, 1]
                    result.push_back((pixel_value / 127.5f) - 1.0f);
                } else {
                    result.push_back(0.0f);
                }
            }
        }
        
        // Cache the result
        preprocessing_cache_.put(cache_key, result);
        
        // Update statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        double processing_time = std::chrono::duration<double>(end_time - start_time).count();
        
        stats_.frames_processed++;
        stats_.total_processing_time.fetch_add(processing_time);
        
        return result;
    }
};

/**
 * Auto-scaling system monitor
 */
class AutoScaler {
public:
    struct ScalingPolicy {
        double cpu_threshold_up = 70.0;    // Scale up if CPU > 70%
        double cpu_threshold_down = 30.0;  // Scale down if CPU < 30%
        size_t min_threads = 2;
        size_t max_threads = std::thread::hardware_concurrency() * 2;
        std::chrono::seconds evaluation_interval{5};
    };

private:
    ScalingPolicy policy_;
    std::atomic<size_t> current_threads_{4};
    std::thread monitoring_thread_;
    std::atomic<bool> monitoring_active_{false};
    
    std::function<void(size_t)> scale_callback_;

public:
    AutoScaler(ScalingPolicy policy = ScalingPolicy{}) : policy_(policy) {}
    
    void set_scaling_callback(std::function<void(size_t)> callback) {
        scale_callback_ = callback;
    }
    
    void start_monitoring() {
        if (monitoring_active_.load()) return;
        
        monitoring_active_ = true;
        monitoring_thread_ = std::thread([this] {
            monitoring_loop();
        });
    }
    
    void stop_monitoring() {
        monitoring_active_ = false;
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
    }
    
    size_t get_current_threads() const {
        return current_threads_.load();
    }

private:
    void monitoring_loop() {
        while (monitoring_active_.load()) {
            // Simulate CPU usage monitoring (in real implementation, use system APIs)
            double cpu_usage = get_cpu_usage();
            size_t current = current_threads_.load();
            size_t new_count = current;
            
            if (cpu_usage > policy_.cpu_threshold_up && current < policy_.max_threads) {
                new_count = std::min(current + 1, policy_.max_threads);
            } else if (cpu_usage < policy_.cpu_threshold_down && current > policy_.min_threads) {
                new_count = std::max(current - 1, policy_.min_threads);
            }
            
            if (new_count != current) {
                current_threads_ = new_count;
                if (scale_callback_) {
                    scale_callback_(new_count);
                }
            }
            
            std::this_thread::sleep_for(policy_.evaluation_interval);
        }
    }
    
    double get_cpu_usage() const {
        // Simplified CPU usage simulation
        // In real implementation, use platform-specific APIs
        return 40.0 + (rand() % 40);  // 40-80% simulated CPU usage
    }
};

} // namespace LiquidVision