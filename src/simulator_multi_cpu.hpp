#pragma once

#include "../packages/plf_nanotimer/plf_nanotimer.hpp"

#include "simulator.hpp"
#include "util.hpp"

// CPUSimulatorSingle represents a single-core CPU particle physics simulator using AVX vector instructions
class SimulatorMultiCPU: public Simulator
{
public:
  int Init(int particles, int area_width, int area_height);
  float* Update(float gravity);

private:
  int count = 0;
  plf::nanotimer timer;

  float* positions_x = nullptr;
  float* positions_y = nullptr;
  float* velocities_x = nullptr;
  float* velocities_y = nullptr;
};
