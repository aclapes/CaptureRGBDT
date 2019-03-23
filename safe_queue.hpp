//
//  safe_queue.hpp
//  a template for an async queue
//  https://stackoverflow.com/questions/15278343/c11-thread-safe-queue
//
//  Created by ChewOnThis_Trident on 13/04/2018.
//  peek method added by aclapes.
//

#ifndef SAFE_QUEUE
#define SAFE_QUEUE

#include <queue>
#include <mutex>
#include <condition_variable>

// A threadsafe-queue.
template <class T>
class SafeQueue
{
public:
    SafeQueue(void)
    : q()
    , m()
    , c_peek()
    , c_dequeue()
    {}
    
    ~SafeQueue(void)
    {}
    
    // Add an element to the queue.
    void enqueue(T t)
    {
        std::lock_guard<std::mutex> lock(m);
        q.push(t);
        c_peek.notify_one();
        c_dequeue.notify_one();
    }

    T peek(int timeout_ms = 500)
    {
        std::unique_lock<std::mutex> lock(m);
        if ( c_peek.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this](){return !q.empty();}) ) 
        {
            T val = q.front();
            return val;
        }
        else
        {
            throw std::runtime_error("Empty queue, no elements to peek.");
        }
        
    }
    
    // Get the "front"-element.
    // If the queue is empty, wait till a element is avaiable.
    T dequeue(void)
    {
        std::unique_lock<std::mutex> lock(m);
        while(q.empty())
        {
            // release lock as long as the wait and reaquire it afterwards.
            c_dequeue.wait(lock);
        }
        T val = q.front();
        q.pop();
        return val;
    }
    
    size_t size(void)
    {
        size_t size_val;
        
        std::lock_guard<std::mutex> lock(m);
        size_val = q.size();
        // c_peek.notify_one();
        // c_dequeue.notify_one();
        
        return size_val;
    }
    
private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c_dequeue;
    std::condition_variable c_peek;
};

// template <class T>
// class SafeDoubleQueue
// {
// public:
//     SafeDoubleQueue(void)
//     : q1()
//     , m1(),

//     , c_peek()
//     , c_dequeue()
//     {}
    
//     ~SafeQueue(void)
//     {}
    
//     // Add an element to the queue.
//     void enqueue(T t)
//     {
//         std::lock_guard<std::mutex> lock(m);
//         q.push(t);
//         c_peek.notify_one();
//         c_dequeue.notify_one();
//     }

//     T peek(int timeout_ms = 500)
//     {
//         std::unique_lock<std::mutex> lock(m);
//         if ( c_peek.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this](){return !q.empty();}) ) 
//         {
//             T val = q.front().clone();
//             return val;
//         }
//         else
//         {
//             throw std::runtime_error("Empty queue, no elements to peek.");
//         }
        
//     }
    
//     // Get the "front"-element.
//     // If the queue is empty, wait till a element is avaiable.
//     T dequeue(void)
//     {
//         std::unique_lock<std::mutex> lock(m);
//         while(q.empty())
//         {
//             // release lock as long as the wait and reaquire it afterwards.
//             c_dequeue.wait(lock);
//         }
//         T val = q.front();
//         q.pop();
//         return val;
//     }
    
//     size_t size(void)
//     {
//         size_t size_val;
        
//         std::lock_guard<std::mutex> lock(m);
//         size_val = q.size();
//         // c_peek.notify_one();
//         // c_dequeue.notify_one();
        
//         return size_val;
//     }
    
// private:
//     std::queue<T> q;
//     mutable std::mutex m;
//     std::condition_variable c_dequeue;
//     std::condition_variable c_peek;
// };
#endif /* safe_queue_h */
