#pragma once

#include "simulator.hpp"

// CPUSimulatorSingle represents a single-core CPU particle physics simulator
class SimulatorCPU : public Simulator
{
public:
  void Init(int particles, int area_width, int area_height)
  {
    count = particles;

    // Init particles
    srand(2277);

    positions = new vec2f[count];
    velocities = new vec2f[count];

    for (unsigned i = 0; i < particles; i++)
    {
      float x = rand() % area_width;
      float y = rand() % area_height;
      positions[i] = { x, y };
    }
  }

  vec2f* Update(float gravity)
  {
    // Compute velocities
    for (unsigned i = 0; i < count; i++)
    for (unsigned j = 0; j < count; j++)
    {
      vec2f diff = positions[j] - positions[i];
      float mult = gravity * diff.len();
      velocities[i] += mult * diff;
    }

    // Compute positions
    for (unsigned i = 0; i < count; i++)
    {
      positions[i] += velocities[i];
    }

    return positions;
  }

private:
  int count = 0;

  vec2f* positions = nullptr;
  vec2f* velocities = nullptr;
};
