#pragma once

#include <cuda_runtime.h>

#include "../packages/plf_nanotimer/plf_nanotimer.hpp"

#include "simulator.hpp"
#include "util.hpp"

__global__ void update_kernel();

// CPUSimulatorSingle represents a single-core CPU particle physics simulator using AVX vector instructions
class SimulatorGPU : public Simulator
{
public:
  int Init(int particles, int area_width, int area_height);
  float* Update(float gravity);

private:
  int count = 0;
  plf::nanotimer timer;

  float* positions_velocities_GPU = nullptr;
  float* positions_velocities = nullptr;

};
