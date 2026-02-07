# Convolution Accelerator with If-Else Branches

A high-performance convolution accelerator implementation featuring clearly defined conditional branching (if-else structures) for flexible configuration and robust edge case handling.

## Overview

This project implements the `std_conv()` function, a critical component of a convolution accelerator, with comprehensive if-else branching to handle:

1. **Boundary Condition Checks** - Logical checks for managing block processing boundaries
2. **Activation Function Selection** - Dynamic selection between ReLU, Leaky ReLU, and Sigmoid
3. **Ping-Pong Buffer Switching** - Efficient double-buffering for input/output processing
4. **Padding Edge Cases** - Robust handling of input feature map padding scenarios

## Features

### 1. Block Processing with Boundary Checks

The implementation processes convolutions in configurable blocks for better cache locality. If-else branches detect and handle boundary blocks that don't align perfectly with block size:

```cpp
// Boundary condition checks for block processing
if (block_end_h < block_start_h + config.block_size || 
    block_end_w < block_start_w + config.block_size) {
    // Boundary block - process with special care
} else {
    // Full block - can optimize processing
}
```

### 2. Flexible Activation Functions

The implementation supports multiple activation functions with clear if-else branching:

- **ReLU**: `f(x) = max(0, x)`
- **Leaky ReLU**: `f(x) = x if x > 0, else α*x`
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
- **None**: Linear pass-through

```cpp
// Activation function selection
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
```

### 3. Ping-Pong Buffering

Efficient double-buffering mechanism with if-else branches for buffer selection:

```cpp
// Ping-pong buffer switching
if (config.use_ping_pong) {
    // Write to current output buffer
    if (current_output_buffer == BUFFER_A) {
        buffer_a[output_idx] = activated_value;
    } else {
        buffer_b[output_idx] = activated_value;
    }
} else {
    // Direct write to output
    output[output_idx] = activated_value;
}
```

### 4. Zero-Padding Edge Case Handling

Robust boundary checking for padding with clear if-else logic:

```cpp
// Edge case handling for padding
if (ih >= 0 && ih < config.input_height && 
    iw >= 0 && iw < config.input_width) {
    // Valid input position - use actual input value
    sum += input[input_idx] * weights[weight_idx];
} else {
    // Outside input bounds - use padding value (zero)
}
```

## Building and Testing

### Prerequisites

- C++ compiler with C++11 support (g++, clang++, etc.)
- Make

### Build

```bash
make
```

### Run Tests

```bash
make test
```

### Clean

```bash
make clean
```

## Test Suite

The implementation includes comprehensive tests covering:

1. **Basic Convolution** - Validates core convolution operation
2. **ReLU Activation** - Tests ReLU branch with negative value zeroing
3. **Leaky ReLU Activation** - Tests leaky ReLU with configurable alpha
4. **Sigmoid Activation** - Tests sigmoid with output range [0,1]
5. **Padding Edge Cases** - Tests boundary handling with padding
6. **Ping-Pong Buffers** - Tests double-buffering mechanism
7. **Block Boundaries** - Tests non-aligned block processing
8. **Multiple Channels** - Tests multi-channel convolutions

## Configuration

The `ConvConfig` structure allows flexible configuration:

```cpp
ConvConfig config;
config.input_height = 32;
config.input_width = 32;
config.input_channels = 3;
config.output_channels = 64;
config.kernel_size = 3;
config.stride = 1;
config.padding = 1;
config.activation = RELU;
config.leaky_relu_alpha = 0.01f;
config.block_size = 8;
config.use_ping_pong = true;
config.debug_mode = false;  // Set to true for debug output
```

## Code Structure

- `conv_accelerator.h` - Header file with class definitions and enums
- `conv_accelerator.cpp` - Implementation of std_conv() with if-else branches
- `test_conv_accelerator.cpp` - Comprehensive test suite
- `Makefile` - Build configuration

## If-Else Branch Summary

The implementation includes **4 major categories** of if-else branches:

1. **Boundary Conditions** (Block Processing)
   - Full vs. boundary block detection
   - Block dimension validation

2. **Activation Functions**
   - ReLU branches (positive/negative check)
   - Leaky ReLU branches (with alpha scaling)
   - Sigmoid computation
   - No-activation pass-through

3. **Buffer Management**
   - Ping-pong buffer selection (BUFFER_A vs BUFFER_B)
   - Buffer switching logic
   - Direct output vs. buffered output

4. **Padding Edge Cases**
   - Valid input coordinate checking
   - Zero-padding for out-of-bounds access

## Benefits

- **Maintainability**: Clear conditional logic makes code easier to understand and modify
- **Flexibility**: Easy to add new activation functions or processing modes
- **Robustness**: Explicit edge case handling prevents undefined behavior
- **Configurability**: Runtime configuration without recompilation

## License

MIT License