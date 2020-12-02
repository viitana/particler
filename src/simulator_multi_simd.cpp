#include "simulator_multi_simd.hpp"

int SimulatorMultiSIMD::Init(int particles, int area_width, int area_height)
{
  count = util::round_up(particles, 8);

  // Init particles
  srand(2277);

  positions_x = new float[2 * count];
  positions_y = positions_x + count;
  velocities_x = new float[count];
  velocities_y = new float[count];

  util::init_particles_circle(
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

float* SimulatorMultiSIMD::Update(float gravity)
{
  double start = timer.get_elapsed_ns();

  float gravity_mult = 1.f / gravity;

  const __m256 v_gravity = _mm256_set1_ps(gravity);

  float velocity_add_x[8];
  float velocity_add_y[8];

  #pragma omp parallel for
  for (int i = 0; i < count; i++)
  {
    // Velocity additions for i
    __m256 v_new_velocities_x = _mm256_set1_ps(0.f);
    __m256 v_new_velocities_y = _mm256_set1_ps(0.f);

    // i position
    __m256 v_position_xi = _mm256_set1_ps(positions_x[i]);
    __m256 v_position_yi = _mm256_set1_ps(positions_y[i]);

    for (int j = 0; j < count; j+=8)
    {
      // Skip vectorization when i == j
      if (i - j >= 0 && i - j < 8)
      {
        for (int k = 0; k < 8; k++)
        {
          if (j + k == i) continue;

          float diff_x = positions_x[j + k] - positions_x[i];
          float diff_y = positions_y[j + k] - positions_y[i];

          float diff_len_sq = diff_x * diff_x + diff_y * diff_y;
          float mult = gravity / diff_len_sq;

          velocities_x[i] += mult * diff_x;
          velocities_y[i] += mult * diff_y;
        }
      }
      else
      {
        __m256 v_position_xj = _mm256_loadu_ps(&positions_x[j]);
        __m256 v_position_yj = _mm256_loadu_ps(&positions_y[j]);

        __m256 v_position_delta_x = _mm256_sub_ps(v_position_xj, v_position_xi);
        __m256 v_position_delta_y = _mm256_sub_ps(v_position_yj, v_position_yi);

        // square dist =  pos diff x ^2 + pos diff y ^2

        __m256 v_position_sqdelta_x = _mm256_mul_ps(v_position_delta_x, v_position_delta_x);
        __m256 v_position_sqdelta_y = _mm256_mul_ps(v_position_delta_y, v_position_delta_y);

        __m256 v_dist_sq = _mm256_add_ps(v_position_sqdelta_x, v_position_sqdelta_y);

        // mult = gravity / square dist

        __m256 v_velocity_mult = _mm256_div_ps(v_gravity, v_dist_sq);

        // added velocity = pos diff * mult

        __m256 v_vx_add = _mm256_mul_ps(v_position_delta_x, v_velocity_mult);
        __m256 v_vy_add = _mm256_mul_ps(v_position_delta_y, v_velocity_mult);

        // velocity += added velocity

        v_new_velocities_x = _mm256_add_ps(v_new_velocities_x, v_vx_add);
        v_new_velocities_y = _mm256_add_ps(v_new_velocities_y, v_vy_add);
      }
    }

    _mm256_storeu_ps(&velocity_add_x[0], v_new_velocities_x);
    _mm256_storeu_ps(&velocity_add_y[0], v_new_velocities_y);

    for (int j = 0; j < 8; j++)
    {
      velocities_x[i] += velocity_add_x[j];
      velocities_y[i] += velocity_add_y[j];
    }
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
