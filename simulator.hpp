#pragma once

#include <vector>

#include "vec2f.hpp"

// Simulator represents a particle physics simulator or computer
// e.g. a single core CPU simulator, multi-core CPU simulator or GPU simulator
class Simulator
{
public:
  virtual void Init(int particles, int area_width, int area_height) = 0;
  virtual vec2f* Update(float gravity) = 0;
};
