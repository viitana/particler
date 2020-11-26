#pragma once

#include <iostream>
#include <cmath>

struct vec2f
{
  float x,y;
  vec2f(float x_, float y_) : x(x_), y(y_) {}
  vec2f() : x(0), y(0) {}

  vec2f operator+(const vec2f& other)
  {
    return {x + other.x , y + other.y};
  }

  void operator+=(const vec2f& other)
  {
    x += other.x;
    y += other.y;
  }

  vec2f operator-(const vec2f& other)
  {
    return {x - other.x , y - other.y};
  }

  void operator-=(const vec2f& other)
  {
    x -= other.x;
    y -= other.y;
  }

  vec2f operator*(const vec2f& other)
  {
    return {x * other.x , y * other.y};
  }

  float len()
  {
    return sqrtf(x*x + y*y);
  }

  float distTo(const vec2f& other)
  {
    return (*this - other).len();
  }

  friend std::ostream& operator<< (std::ostream &out, const vec2f& v);
};

std::ostream& operator<< (std::ostream &out, const vec2f& v)
{
    return out << "vec3f(" << v.x << ", " << v.y << ")";
}

vec2f operator*(const float f, const vec2f& v)
{
  return {f * v.x , f * v.y};
}

vec2f operator*(const vec2f& v, const float f)
{
  return f * v;
}

vec2f operator/(const float f, const vec2f& v)
{
  return {f / v.x , f / v.y};
}

vec2f operator/(const vec2f& v, const float f)
{
  return {v.x / f , v.y / f};
}
