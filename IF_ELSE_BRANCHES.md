# If-Else Branch Implementation Summary

This document provides a detailed summary of all if-else branches implemented in the `std_conv()` function for the convolution accelerator.

## 1. Boundary Condition Checks for Block Processing

**Location**: `conv_accelerator.cpp`, lines ~70-80

**Purpose**: Detect and handle boundary blocks that don't align with the configured block size.

```cpp
// Check if this is a boundary block
if (block_end_h < block_start_h + config.block_size || 
    block_end_w < block_start_w + config.block_size) {
    // Boundary block - process with special care
    std::cout << "Processing boundary block at (" << block_h << ", " << block_w << ")" << std::endl;
} else {
    // Full block - can optimize processing
    std::cout << "Processing full block at (" << block_h << ", " << block_w << ")" << std::endl;
}
```

**Benefits**:
- Enables block-based processing for better cache locality
- Handles non-aligned dimensions gracefully
- Provides framework for future optimizations (e.g., SIMD for full blocks)

---

## 2. Activation Function Selection

**Location**: `conv_accelerator.cpp`, lines ~125-138

**Purpose**: Apply the appropriate activation function based on configuration.

```cpp
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
```

### 2.1 ReLU Implementation

**Location**: `conv_accelerator.cpp`, lines ~17-22

```cpp
if (type == RELU) {
    // ReLU: max(0, x)
    if (value > 0.0f) {
        return value;
    } else {
        return 0.0f;
    }
}
```

### 2.2 Leaky ReLU Implementation

**Location**: `conv_accelerator.cpp`, lines ~23-29

```cpp
else if (type == LEAKY_RELU) {
    // Leaky ReLU: x if x > 0, else alpha * x
    if (value > 0.0f) {
        return value;
    } else {
        return alpha * value;
    }
}
```

### 2.3 Sigmoid Implementation

**Location**: `conv_accelerator.cpp`, lines ~30-33

```cpp
else if (type == SIGMOID) {
    // Sigmoid: 1 / (1 + exp(-x))
    return 1.0f / (1.0f + std::exp(-value));
}
```

**Benefits**:
- Flexible activation function selection at runtime
- Easy to add new activation functions
- No recompilation needed for configuration changes

---

## 3. Ping-Pong Buffer Switching

**Location**: `conv_accelerator.cpp`, lines ~142-156

**Purpose**: Enable efficient double-buffering for concurrent read/write operations.

### 3.1 Buffer Selection for Writing

```cpp
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
```

### 3.2 Buffer Selection for Reading

**Location**: `conv_accelerator.cpp`, lines ~41-47

```cpp
std::vector<float>& ConvAccelerator::get_buffer(BufferSelect buffer) {
    // If-else branch for buffer selection
    if (buffer == BUFFER_A) {
        return buffer_a;
    } else {
        return buffer_b;
    }
}
```

### 3.3 Buffer Switching Logic

**Location**: `conv_accelerator.cpp`, lines ~166-175

```cpp
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
```

**Benefits**:
- Enables concurrent processing in hardware accelerators
- Reduces memory bottlenecks
- Framework for pipeline optimization

---

## 4. Input Feature Map Padding Edge Cases

**Location**: `conv_accelerator.cpp`, lines ~106-117

**Purpose**: Handle out-of-bounds accesses during convolution with proper zero-padding.

```cpp
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
```

**Benefits**:
- Robust boundary checking prevents out-of-bounds access
- Supports variable padding configurations
- Clear separation of valid vs. padded regions

---

## Test Coverage

Each if-else branch is validated by the test suite:

1. **test_basic_convolution()** - Tests basic flow and block processing
2. **test_relu_activation()** - Tests ReLU branch (positive/negative check)
3. **test_leaky_relu_activation()** - Tests Leaky ReLU with alpha parameter
4. **test_sigmoid_activation()** - Tests Sigmoid activation
5. **test_padding()** - Tests padding edge case handling
6. **test_ping_pong_buffers()** - Tests buffer switching logic
7. **test_block_boundaries()** - Tests boundary block detection
8. **test_multiple_channels()** - Tests multi-channel processing

All tests pass successfully, validating the correctness of all if-else branches.

---

## Future Extension Points

The if-else structure provides clear extension points for:

1. **Additional Activation Functions**: Add new cases to activation selection
2. **Hardware-Specific Optimizations**: Add branches for SIMD, GPU, or FPGA paths
3. **Dynamic Block Sizing**: Add branches for adaptive block size selection
4. **Advanced Padding Modes**: Extend padding branch for reflect/symmetric modes
5. **Multi-Buffer Support**: Extend ping-pong to support N-way buffering

---

## Summary Statistics

- **Total If-Else Branches**: 15
- **Categories**: 4 major categories
- **Lines of Conditional Code**: ~60 lines
- **Test Coverage**: 8 comprehensive tests
- **All Tests Status**: ✓ PASSING
