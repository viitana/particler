#pragma once

// Simulator represents a particle physics simulator or computer
// e.g. a single core CPU simulator, multi-core CPU simulator or GPU simulator
class Simulator
{
public:
  virtual int Init(int particles, int area_width, int area_height) = 0;
  virtual float* Update(float gravity) = 0;
};
