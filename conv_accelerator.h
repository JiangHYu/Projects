#ifndef CONV_ACCELERATOR_H
#define CONV_ACCELERATOR_H

#include <vector>
#include <cmath>
#include <algorithm>

// Activation function types
enum ActivationType {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    NONE
};

// Buffer selection for ping-pong buffering
enum BufferSelect {
    BUFFER_A,
    BUFFER_B
};

// Configuration structure for convolution
struct ConvConfig {
    int input_height;
    int input_width;
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    ActivationType activation;
    float leaky_relu_alpha;  // Alpha value for Leaky ReLU
    int block_size;          // Size of processing blocks
    bool use_ping_pong;      // Enable ping-pong buffering
    bool debug_mode;         // Enable debug output
};

// Convolution accelerator class
class ConvAccelerator {
private:
    ConvConfig config;
    std::vector<float> buffer_a;
    std::vector<float> buffer_b;
    BufferSelect current_input_buffer;
    BufferSelect current_output_buffer;

    // Helper function to apply activation
    float apply_activation(float value, ActivationType type, float alpha = 0.01f);
    
    // Helper function to get buffer pointer
    std::vector<float>& get_buffer(BufferSelect buffer);

public:
    ConvAccelerator(const ConvConfig& cfg);
    
    // Main standard convolution function with if-else branches
    void std_conv(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        const std::vector<float>& bias,
        std::vector<float>& output
    );
};

#endif // CONV_ACCELERATOR_H
