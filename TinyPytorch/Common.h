#pragma once

// Common definitions shared between CPU and CUDA tensor headers.
enum class DataType
{
	float32,
	int32
};

// Forward declarations
class Tensor;
class CPUTensor;
