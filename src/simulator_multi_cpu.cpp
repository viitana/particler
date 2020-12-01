#include "simulator_multi_cpu.hpp"

#include <stdlib.h>

// CPUSimulatorSingle represents a single-core CPU particle physics simulator using AVX vector instructions
int SimulatorMultiCPU::Init(int particles, int area_width, int area_height)
{
  count = particles;

  // Init particles
  srand(2277);

  positions_x = new float[2 * count];
  positions_y = positions_x + count;
  velocities_x = new float[count];
  velocities_y = new float[count];

  util::init_particles_random(
    positions_x,
    positions_y,
    velocities_x,
    velocities_y,
    count,
    area_width,
    area_height
  );

  return count;
}

float* SimulatorMultiCPU::Update(float gravity)
{
  double start = timer.get_elapsed_ns();

  // Compute velocities
  #pragma omp parallel for
  for (int i = 0; i < count; i++)
  for (int j = 0; j < count; j++)
  {
    if (i == j) continue;

    float diff_x = positions_x[j] - positions_x[i];
    float diff_y = positions_y[j] - positions_y[i];

    float diff_len_sq = diff_x * diff_x + diff_y * diff_y;
    float mult = gravity / diff_len_sq;

    velocities_x[i] += mult * diff_x;
    velocities_y[i] += mult * diff_y;
  }

  double timestep = (timer.get_elapsed_ns() - start) * 0.0000001;

  // Compute positions
  #pragma omp parallel for
  for (int i = 0; i < count; i++)
  {
    positions_x[i] += velocities_x[i] * timestep;
    positions_y[i] += velocities_y[i] * timestep;
  }

  return positions_x;
}
