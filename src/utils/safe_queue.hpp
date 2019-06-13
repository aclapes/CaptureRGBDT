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

namespace uls
{
    // A threadsafe-queue.
    // template <class T>
    // class SafeQueue
    // {
    // public:
    //     SafeQueue(void)
    //     : q()
    //     , m()
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
    //             T val = q.front();
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

    // template <class T>
    // class SafeQueue
    // {
    // public:
    //     SafeQueue(int max_buffer_size = 2)
    //     : q()
    //     , m()
    //     , c_peek()
    //     , c_dequeue()
    //     , max_buffer_size(max_buffer_size)
    //     , min_buffer_size(2)
    //     {
    //     }
        
    //     ~SafeQueue(void)
    //     {}
        
    //     // Add an element to the queue.
    //     void enqueue(T t)
    //     {
    //         std::unique_lock<std::mutex> lock(m);
    //         while(q.size() >= this->max_buffer_size)
    //         {
    //             c_enqueue.wait(lock);
    //         }
    //         q.push(t);
    //         c_peek.notify_one();
    //         c_dequeue.notify_one();
    //     }

    //     T peek(int timeout_ms = 500)
    //     {
    //         std::unique_lock<std::mutex> lock(m);
    //         if ( c_peek.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this](){return !q.empty();}) ) 
    //         {
    //             T val = q.front();
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
    //         while(q.size() < this->min_buffer_size)
    //         {
    //             // release lock as long as the wait and reaquire it afterwards.
    //             c_dequeue.wait(lock);
    //         }

    //         T val = q.front();
    //         q.pop();

    //         if (q.size() == this->min_buffer_size)
    //         {
    //             c_enqueue.notify_one();
    //         }

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
    //     std::condition_variable c_enqueue;
    //     std::condition_variable c_peek;

    //     int min_buffer_size;
    //     int max_buffer_size;
    // };

    class dequeue_error : public std::runtime_error
    {
        public:
            dequeue_error() : std::runtime_error("") {}
            dequeue_error(const char* what_arg) : std::runtime_error(what_arg) {}
            dequeue_error(const std::string what_arg) : std::runtime_error(what_arg) {}
    };

    class peek_error : public std::runtime_error
    {
        public:
            peek_error() : std::runtime_error("") {}
            peek_error(const char* what_arg) : std::runtime_error(what_arg) {}
            peek_error(const std::string what_arg) : std::runtime_error(what_arg) {}
    };

    template <class T>
    class SafeQueue
    {
    public:
        SafeQueue()
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
            std::unique_lock<std::mutex> lock(m);
            q.push(t);
            c_peek.notify_one();
            c_dequeue.notify_one();
        }

        // T peek(int timeout_ms = 500)
        // {
        //     std::unique_lock<std::mutex> lock(m);
        //     if ( c_peek.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this](){ return !q.empty(); }) ) 
        //     {
        //         T val = q.front();
        //         return val;
        //     }
        //     else
        //     {
        //         throw std::runtime_error("Empty queue, no elements to peek.");
        //     }
        // }

        T peek_back()
        {
            std::unique_lock<std::mutex> lock(m);

            if (q.empty())
                throw peek_error();

            return q.back();
        }

        T peek_front()
        {
            std::unique_lock<std::mutex> lock(m);

            if (q.empty())
                throw peek_error();

            return q.front();
        }

        T dequeue(int timeout_ms = 2500)
        {
            std::unique_lock<std::mutex> lock(m);

            if ( 
                c_dequeue.wait_for( 
                    lock, 
                    std::chrono::milliseconds(timeout_ms), 
                    [this](){ 
                        return q.size() > 0; // do not dequeue 
                    }
                ) 
            )
            {
                T val = q.front();
                q.pop();
                return val;
            }
            else
            {
                throw dequeue_error();
            }
            
        }

        int get_min_buffer_size()
        {
            return this->min_buffer_size;
        }
        
        // Get the "front"-element.
        // If the queue is empty, wait till a element is avaiable.
        // T dequeue(void)
        // {
        //     std::unique_lock<std::mutex> lock(m);
        //     while(q.size() < this->min_buffer_size)
        //     {
        //         // release lock as long as the wait and reaquire it afterwards.
        //         c_dequeue.wait(lock);
        //     }

        //     T val = q.front();
        //     q.pop();

        //     return val;
        // }
        
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

        int min_buffer_size;
    };
}

#endif /* safe_queue_h */
