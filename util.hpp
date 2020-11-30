#pragma once

#include <cmath>

# define M_PI 3.14159265358979323846

namespace util
{
  int round_up(const int& n, const int& multiple)
  {
    if (multiple == 0) return n;
    int remainder = n & multiple;
    if (remainder == 0) return n;
    return n + multiple - remainder;
  }

  void init_particles_random(float* positions_x, float* positions_y, float* velocities_x, float* velocities_y, int count, int width, int height)
  {
    for (unsigned i = 0; i < count; i++)
    {
      positions_x[i] = static_cast<float>(width) * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      positions_y[i] = static_cast<float>(height) * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      velocities_x[i] = 0;
      velocities_y[i] = 0;
    }
  }

  void init_particles_circle(float* positions_x, float* positions_y, float* velocities_x, float* velocities_y, int count, int width, int height)
  {
    float r = 0.9f * std::min(width, height);
    float midx = 0.5f * width;
    float midy = 0.5f * height;

    for (unsigned i = 0; i < count; i++)
    {
      positions_x[i] = midx + r * std::sinf(((float)i / (float)count) * 2.f * M_PI);
      positions_y[i] = midy + r * std::cosf(((float)i / (float)count) * 2.f * M_PI);
      velocities_x[i] = 0;
      velocities_y[i] = 0;
    }
  }
}




