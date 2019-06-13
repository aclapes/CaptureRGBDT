//
//  pt/pipeline.hpp
//  RealsenseExamplesGettingStarted
//
//  Created by Albert Clapés on 27/11/2018.
//

#ifndef pt_pipeline_h
#define pt_pipeline_h

#define PT1_VID 0x1e4e
#define PT1_PID 0x0100

#include <exception>
#include "libuvc/libuvc.h"

namespace pt
{
    class config
    {
    private:
        int height;
        int width;

    public:
        config() : height(120), width(160)
        {

        }

        void set_stream(int w, int h)
        {
            height = h;
            width = w;
        }

        int get_width()
        {
            return width;
        }

        int get_height()
        {
            return height;
        }
    };

    class pipeline
    {
    private:
        uvc_context_t *ctx = NULL;
        uvc_device_t *dev = NULL;
        uvc_device_handle_t *devh;
        uvc_stream_handle_t *strmh;
        uvc_stream_ctrl_t ctrl;
        uvc_frame_t **frame;
        
    public:
        pipeline()
        {
            // uvc_error_t res;
            
            // res = uvc_init(&(this->ctx), NULL);
            // if (res < 0) {
            //     uvc_perror(res, "uvc_init");
            //     throw std::runtime_error("uvc_init failed");
            // }
            
            // /* filter devices: vendor_id, product_id, "serial_num" */
            // res = uvc_find_device(this->ctx, &(this->dev), PT1_VID, PT1_PID, NULL);
            // if (res < 0) {
            //     uvc_perror(res, "uvc_find_device"); /* no devices found */
            //     throw std::runtime_error("uvc_find_device failed");
            // }
            
            // /* Try to open the device: requires exclusive access */
            // res = uvc_open(dev, &devh);
            // if (res < 0) {
            //     uvc_perror(res, "uvc_open"); /* unable to open device */
            //     /* Release the device descriptor */
            //     uvc_unref_device(dev);
            //     throw std::runtime_error("uvc_open failed");
            // }
            
            // /* Print out a message containing all the information that libuvc
            //  * knows about the device */
            // uvc_print_diag(devh, stderr);
            
            // /* Try to negotiate a 160x120 9 fps Y16 stream profile */
            // res = uvc_get_stream_ctrl_format_size(
            //                                       devh, &ctrl, /* result stored in ctrl */
            //                                       UVC_FRAME_FORMAT_Y16, /* YUV 422, aka YUV 4:2:2. try _COMPRESSED */
            //                                       160, 120, 9 /* width, height, fps */
            //                                       );
            // uvc_print_stream_ctrl(&ctrl, stderr);
            // if (res < 0) {
            //     uvc_perror(res, "get_mode");
            //     throw std::runtime_error("get_stream_ctrl_format_size failed");
            // }
            
            // res = uvc_stream_open_ctrl(devh, &strmh, &ctrl);
            // if (res < 0) {
            //     uvc_perror(res, "stream_open_ctrl");
            //     throw std::runtime_error("stream_open_ctrl failed");
            // }
        }
        
        ~pipeline()
        {
            // uvc_close(devh);
            // uvc_unref_device(dev);
            // uvc_exit(ctx);
        }

        void start()
        {
            pt::config cfg;
            start(cfg);
        }
        
        void start(pt::config cfg)
        {
            uvc_error_t res;

            res = uvc_init(&(this->ctx), NULL);
            if (res < 0) {
                uvc_perror(res, "uvc_init");
                throw std::runtime_error("uvc_init failed");
            }
            
            /* filter devices: vendor_id, product_id, "serial_num" */
            res = uvc_find_device(this->ctx, &(this->dev), PT1_VID, PT1_PID, NULL);
            if (res < 0) {
                uvc_perror(res, "uvc_find_device"); /* no devices found */
                throw std::runtime_error("uvc_find_device failed");
            }
            
            /* Try to open the device: requires exclusive access */
            res = uvc_open(dev, &devh);
            if (res < 0) {
                uvc_perror(res, "uvc_open"); /* unable to open device */
                /* Release the device descriptor */
                uvc_unref_device(dev);
                throw std::runtime_error("uvc_open failed");
            }
            
            /* Print out a message containing all the information that libuvc
             * knows about the device */
            uvc_print_diag(devh, stderr);
            
            /* Try to negotiate a 160x120 9 fps Y16 stream profile */
            res = uvc_get_stream_ctrl_format_size(
                                                  devh, &ctrl, /* result stored in ctrl */
                                                  UVC_FRAME_FORMAT_Y16, /* YUV 422, aka YUV 4:2:2. try _COMPRESSED */
                                                //   160, 120, 9 /* width, height, fps */
                                                  cfg.get_width(), cfg.get_height(), 9
                                                  );
            uvc_print_stream_ctrl(&ctrl, stderr);
            if (res < 0) {
                uvc_perror(res, "get_mode");
                throw std::runtime_error("get_stream_ctrl_format_size failed");
            }
            
            res = uvc_stream_open_ctrl(devh, &strmh, &ctrl);
            if (res < 0) {
                uvc_perror(res, "stream_open_ctrl");
                throw std::runtime_error("stream_open_ctrl failed");
            }
            
            res = uvc_stream_start(this->strmh, NULL, (void*) NULL, 0, 0);
            if (res < 0)
            {
                uvc_perror(res, "uvc_stream_start"); /* unable to start stream */
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
        void stop()
        {
            uvc_error_t res;
            
            res = uvc_stream_stop(this->strmh);
            if (res < 0)
            {
                uvc_perror(res, "uvc_stream_stop"); /* unable to start stream */
            }
            
            uvc_stop_streaming(devh);

            uvc_close(devh);
            uvc_unref_device(dev);
            uvc_exit(ctx);
        }
        
        uvc_frame_t* wait_for_frames(unsigned int timeout_ms = 0) const
        {
            uvc_frame_t *frame;
            uvc_error_t res;

            res = uvc_stream_get_frame(strmh, &frame, timeout_ms);
            if (res < 0)
            {
                uvc_perror(res, "stream_get_frame");
            }

            return frame;
        }
    };
}

#endif /* pt_pipeline_h */
