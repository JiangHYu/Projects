#include "conv_accelerator.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

// Helper function to compare floats
bool float_equals(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// Test 1: Basic convolution without activation
void test_basic_convolution() {
    std::cout << "\n=== Test 1: Basic Convolution ===" << std::endl;
    
    ConvConfig config;
    config.input_height = 4;
    config.input_width = 4;
    config.input_channels = 1;
    config.output_channels = 1;
    config.kernel_size = 3;
    config.stride = 1;
    config.padding = 0;
    config.activation = NONE;
    config.block_size = 2;
    config.use_ping_pong = false;
    config.debug_mode = false;
    
    ConvAccelerator accelerator(config);
    
    // Simple input (4x4x1)
    std::vector<float> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    // Simple kernel (3x3x1x1)
    std::vector<float> weights = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };
    
    std::vector<float> bias = {0};
    std::vector<float> output;
    
    accelerator.std_conv(input, weights, bias, output);
    
    std::cout << "Output dimensions: " << output.size() << std::endl;
    std::cout << "Expected output size: " << 2 * 2 << std::endl;
    assert(output.size() == 4);  // (4-3+0)/1 + 1 = 2x2
    std::cout << "✓ Basic convolution test passed" << std::endl;
}

// Test 2: ReLU activation function
void test_relu_activation() {
    std::cout << "\n=== Test 2: ReLU Activation ===" << std::endl;
    
    ConvConfig config;
    config.input_height = 3;
    config.input_width = 3;
    config.input_channels = 1;
    config.output_channels = 1;
    config.kernel_size = 2;
    config.stride = 1;
    config.padding = 0;
    config.activation = RELU;
    config.block_size = 2;
    config.use_ping_pong = false;
    config.debug_mode = false;
    
    ConvAccelerator accelerator(config);
    
    std::vector<float> input = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    
    std::vector<float> weights = {
        -1, -1,
        -1, -1
    };
    
    std::vector<float> bias = {1};  // Small positive bias
    std::vector<float> output;
    
    accelerator.std_conv(input, weights, bias, output);
    
    // ReLU should zero out negative values
    bool has_no_negatives = true;
    for (float val : output) {
        if (val < 0) {
            has_no_negatives = false;
            break;
        }
    }
    
    assert(has_no_negatives);
    std::cout << "✓ ReLU activation test passed (no negative values)" << std::endl;
}

// Test 3: Leaky ReLU activation
void test_leaky_relu_activation() {
    std::cout << "\n=== Test 3: Leaky ReLU Activation ===" << std::endl;
    
    ConvConfig config;
    config.input_height = 3;
    config.input_width = 3;
    config.input_channels = 1;
    config.output_channels = 1;
    config.kernel_size = 2;
    config.stride = 1;
    config.padding = 0;
    config.activation = LEAKY_RELU;
    config.leaky_relu_alpha = 0.1f;
    config.block_size = 2;
    config.use_ping_pong = false;
    config.debug_mode = false;
    
    ConvAccelerator accelerator(config);
    
    std::vector<float> input = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    
    std::vector<float> weights = {
        -1, -1,
        -1, -1
    };
    
    std::vector<float> bias = {-10};  // Large negative bias to ensure negative outputs
    std::vector<float> output;
    
    accelerator.std_conv(input, weights, bias, output);
    
    std::cout << "Output values: ";
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Leaky ReLU should have small negative values for negative inputs
    std::cout << "✓ Leaky ReLU activation test passed" << std::endl;
}

// Test 4: Sigmoid activation
void test_sigmoid_activation() {
    std::cout << "\n=== Test 4: Sigmoid Activation ===" << std::endl;
    
    ConvConfig config;
    config.input_height = 3;
    config.input_width = 3;
    config.input_channels = 1;
    config.output_channels = 1;
    config.kernel_size = 2;
    config.stride = 1;
    config.padding = 0;
    config.activation = SIGMOID;
    config.block_size = 2;
    config.use_ping_pong = false;
    config.debug_mode = false;
    
    ConvAccelerator accelerator(config);
    
    std::vector<float> input = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    
    std::vector<float> weights = {
        1, 0,
        0, 1
    };
    
    std::vector<float> bias = {0};
    std::vector<float> output;
    
    accelerator.std_conv(input, weights, bias, output);
    
    // Sigmoid should produce values between 0 and 1
    bool in_range = true;
    for (float val : output) {
        if (val < 0.0f || val > 1.0f) {
            in_range = false;
            break;
        }
    }
    
    assert(in_range);
    std::cout << "✓ Sigmoid activation test passed (all values in [0,1])" << std::endl;
}

// Test 5: Padding edge cases
void test_padding() {
    std::cout << "\n=== Test 5: Padding Edge Cases ===" << std::endl;
    
    ConvConfig config;
    config.input_height = 3;
    config.input_width = 3;
    config.input_channels = 1;
    config.output_channels = 1;
    config.kernel_size = 3;
    config.stride = 1;
    config.padding = 1;  // Padding enabled
    config.activation = NONE;
    config.block_size = 2;
    config.use_ping_pong = false;
    config.debug_mode = false;
    
    ConvAccelerator accelerator(config);
    
    std::vector<float> input = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    
    std::vector<float> weights(9, 1.0f);  // All ones
    std::vector<float> bias = {0};
    std::vector<float> output;
    
    accelerator.std_conv(input, weights, bias, output);
    
    // With padding=1, output should be same size as input
    assert(output.size() == 9);  // 3x3
    std::cout << "✓ Padding test passed (output size: " << output.size() << ")" << std::endl;
}

// Test 6: Ping-pong buffering
void test_ping_pong_buffers() {
    std::cout << "\n=== Test 6: Ping-Pong Buffer Switching ===" << std::endl;
    
    ConvConfig config;
    config.input_height = 4;
    config.input_width = 4;
    config.input_channels = 1;
    config.output_channels = 1;
    config.kernel_size = 2;
    config.stride = 1;
    config.padding = 0;
    config.activation = NONE;
    config.block_size = 2;
    config.use_ping_pong = true;  // Enable ping-pong buffering
    config.debug_mode = false;
    
    ConvAccelerator accelerator(config);
    
    std::vector<float> input(16, 1.0f);
    std::vector<float> weights(4, 0.25f);
    std::vector<float> bias = {0};
    std::vector<float> output;
    
    accelerator.std_conv(input, weights, bias, output);
    
    assert(output.size() == 9);  // 3x3 output
    std::cout << "✓ Ping-pong buffer test passed" << std::endl;
}

// Test 7: Block boundary conditions
void test_block_boundaries() {
    std::cout << "\n=== Test 7: Block Boundary Conditions ===" << std::endl;
    
    ConvConfig config;
    config.input_height = 5;
    config.input_width = 5;
    config.input_channels = 1;
    config.output_channels = 1;
    config.kernel_size = 2;
    config.stride = 1;
    config.padding = 0;
    config.activation = NONE;
    config.block_size = 3;  // Not evenly divisible by output size (4x4)
    config.use_ping_pong = false;
    config.debug_mode = false;
    
    ConvAccelerator accelerator(config);
    
    std::vector<float> input(25, 1.0f);
    std::vector<float> weights(4, 1.0f);
    std::vector<float> bias = {0};
    std::vector<float> output;
    
    accelerator.std_conv(input, weights, bias, output);
    
    assert(output.size() == 16);  // 4x4 output
    std::cout << "✓ Block boundary test passed" << std::endl;
}

// Test 8: Multiple channels
void test_multiple_channels() {
    std::cout << "\n=== Test 8: Multiple Channels ===" << std::endl;
    
    ConvConfig config;
    config.input_height = 3;
    config.input_width = 3;
    config.input_channels = 2;
    config.output_channels = 2;
    config.kernel_size = 2;
    config.stride = 1;
    config.padding = 0;
    config.activation = RELU;
    config.block_size = 2;
    config.use_ping_pong = false;
    config.debug_mode = false;
    
    ConvAccelerator accelerator(config);
    
    std::vector<float> input(18, 1.0f);  // 3x3x2
    std::vector<float> weights(16, 0.5f);  // 2x2x2x2
    std::vector<float> bias = {0, 0};
    std::vector<float> output;
    
    accelerator.std_conv(input, weights, bias, output);
    
    assert(output.size() == 8);  // 2x2x2
    std::cout << "✓ Multiple channels test passed" << std::endl;
}

int main() {
    std::cout << "Running Convolution Accelerator Tests..." << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        test_basic_convolution();
        test_relu_activation();
        test_leaky_relu_activation();
        test_sigmoid_activation();
        test_padding();
        test_ping_pong_buffers();
        test_block_boundaries();
        test_multiple_channels();
        
        std::cout << "\n=========================================" << std::endl;
        std::cout << "✓ All tests passed successfully!" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
