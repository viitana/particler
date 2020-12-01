#pragma once

namespace util
{
  inline int min(int a, int b);

  int round_up(const int& n, const int& multiple);

  void init_particles_random(float* positions_x, float* positions_y, float* velocities_x, float* velocities_y, int count, int width, int height);
  void init_particles_circle(float* positions_x, float* positions_y, float* velocities_x, float* velocities_y, int count, int width, int height);

  // static inline void check(cudaError_t err, const char* context);
}
