#include "conv_accelerator.h"
#include <iostream>
#include <stdexcept>

ConvAccelerator::ConvAccelerator(const ConvConfig& cfg) : config(cfg) {
    // Initialize ping-pong buffers if enabled
    if (config.use_ping_pong) {
        int buffer_size = config.input_height * config.input_width * config.input_channels;
        buffer_a.resize(buffer_size, 0.0f);
        buffer_b.resize(buffer_size, 0.0f);
        current_input_buffer = BUFFER_A;
        current_output_buffer = BUFFER_B;
    }
}

float ConvAccelerator::apply_activation(float value, ActivationType type, float alpha) {
    // If-else branch for activation function selection
    if (type == RELU) {
        // ReLU: max(0, x)
        if (value > 0.0f) {
            return value;
        } else {
            return 0.0f;
        }
    } else if (type == LEAKY_RELU) {
        // Leaky ReLU: x if x > 0, else alpha * x
        if (value > 0.0f) {
            return value;
        } else {
            return alpha * value;
        }
    } else if (type == SIGMOID) {
        // Sigmoid: 1 / (1 + exp(-x))
        return 1.0f / (1.0f + std::exp(-value));
    } else {
        // No activation
        return value;
    }
}

std::vector<float>& ConvAccelerator::get_buffer(BufferSelect buffer) {
    // If-else branch for buffer selection
    if (buffer == BUFFER_A) {
        return buffer_a;
    } else {
        return buffer_b;
    }
}

void ConvAccelerator::std_conv(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    std::vector<float>& output
) {
    // Calculate output dimensions
    int output_height = (config.input_height + 2 * config.padding - config.kernel_size) / config.stride + 1;
    int output_width = (config.input_width + 2 * config.padding - config.kernel_size) / config.stride + 1;
    
    // Validate dimensions
    if (output_height <= 0 || output_width <= 0) {
        throw std::invalid_argument("Invalid output dimensions");
    }
    
    // Resize output
    output.resize(output_height * output_width * config.output_channels, 0.0f);
    
    // Process convolution in blocks for better cache locality
    int num_blocks_h = (output_height + config.block_size - 1) / config.block_size;
    int num_blocks_w = (output_width + config.block_size - 1) / config.block_size;
    
    // Iterate through blocks
    for (int block_h = 0; block_h < num_blocks_h; block_h++) {
        for (int block_w = 0; block_w < num_blocks_w; block_w++) {
            
            // If-else branch 1: Boundary condition checks for block processing
            int block_start_h = block_h * config.block_size;
            int block_end_h = std::min(block_start_h + config.block_size, output_height);
            int block_start_w = block_w * config.block_size;
            int block_end_w = std::min(block_start_w + config.block_size, output_width);
            
            // Check if this is a boundary block
            if (config.debug_mode) {
                if (block_end_h < block_start_h + config.block_size || 
                    block_end_w < block_start_w + config.block_size) {
                    // Boundary block - process with special care
                    std::cout << "Processing boundary block at (" << block_h << ", " << block_w << ")" << std::endl;
                } else {
                    // Full block - can optimize processing
                    std::cout << "Processing full block at (" << block_h << ", " << block_w << ")" << std::endl;
                }
            }
            
            // Process each output channel
            for (int oc = 0; oc < config.output_channels; oc++) {
                
                // Process each position in the current block
                for (int oh = block_start_h; oh < block_end_h; oh++) {
                    for (int ow = block_start_w; ow < block_end_w; ow++) {
                        
                        float sum = 0.0f;
                        
                        // Convolve over input channels
                        for (int ic = 0; ic < config.input_channels; ic++) {
                            for (int kh = 0; kh < config.kernel_size; kh++) {
                                for (int kw = 0; kw < config.kernel_size; kw++) {
                                    
                                    int ih = oh * config.stride + kh - config.padding;
                                    int iw = ow * config.stride + kw - config.padding;
                                    
                                    // If-else branch 4: Edge case handling for padding
                                    if (ih >= 0 && ih < config.input_height && 
                                        iw >= 0 && iw < config.input_width) {
                                        // Valid input position - use actual input value
                                        int input_idx = (ih * config.input_width + iw) * config.input_channels + ic;
                                        int weight_idx = ((oc * config.input_channels + ic) * config.kernel_size + kh) * config.kernel_size + kw;
                                        sum += input[input_idx] * weights[weight_idx];
                                    } else {
                                        // Outside input bounds - use padding value (zero)
                                        // No operation needed as we're using zero padding
                                    }
                                }
                            }
                        }
                        
                        // Add bias
                        sum += bias[oc];
                        
                        // If-else branch 2: Apply activation function based on configuration
                        float activated_value;
                        if (config.activation == RELU) {
                            activated_value = apply_activation(sum, RELU);
                        } else if (config.activation == LEAKY_RELU) {
                            activated_value = apply_activation(sum, LEAKY_RELU, config.leaky_relu_alpha);
                        } else if (config.activation == SIGMOID) {
                            activated_value = apply_activation(sum, SIGMOID);
                        } else {
                            // No activation function
                            activated_value = sum;
                        }
                        
                        // If-else branch 3: Ping-pong buffer switching for output
                        int output_idx = (oh * output_width + ow) * config.output_channels + oc;
                        if (config.use_ping_pong) {
                            // Write to current output buffer
                            if (current_output_buffer == BUFFER_A) {
                                if (output_idx >= 0 && static_cast<size_t>(output_idx) < buffer_a.size()) {
                                    buffer_a[output_idx] = activated_value;
                                }
                            } else {
                                if (output_idx >= 0 && static_cast<size_t>(output_idx) < buffer_b.size()) {
                                    buffer_b[output_idx] = activated_value;
                                }
                            }
                        } else {
                            // Direct write to output
                            output[output_idx] = activated_value;
                        }
                    }
                }
            }
        }
    }
    
    // If-else branch 3 (continued): Copy from ping-pong buffer to output if enabled
    if (config.use_ping_pong) {
        std::vector<float>& output_buffer = get_buffer(current_output_buffer);
        std::copy(output_buffer.begin(), 
                  output_buffer.begin() + std::min(output.size(), output_buffer.size()), 
                  output.begin());
        
        // Switch buffers for next iteration
        if (current_input_buffer == BUFFER_A) {
            current_input_buffer = BUFFER_B;
            current_output_buffer = BUFFER_A;
        } else {
            current_input_buffer = BUFFER_A;
            current_output_buffer = BUFFER_B;
        }
    }
}
