#include "simulator_gpu.cuh"

#include <iostream>

#define CUDA_BLOCK_SIZE 256

inline void check(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << context << ": "
          << cudaGetErrorString(err) << std::endl;
      std::exit(EXIT_FAILURE);
  }
}
#define CHECK(x) check(x, #x)

// __global__ void update_kernel()
// {
//   int i = blockIdx.x;
//   int j = threadIdx.x;

//   if (i == j) return;

//   float diff_x = positions_x[j] - positions_x[i];
//   float diff_y = positions_y[j] - positions_y[i];

//   float diff_len_sq = diff_x * diff_x + diff_y * diff_y;
//   float mult = gravity / diff_len_sq;

//   velocities_x[i] += mult * diff_x;
//   velocities_y[i] += mult * diff_y;
// }


int SimulatorGPU::Init(int particles, int area_width, int area_height)
{
  count = util::round_up(particles, CUDA_BLOCK_SIZE);

  // Init particles
  srand(2277);

  // util::init_particles_random(
  //   positions_x,
  //   positions_y,
  //   velocities_x,
  //   velocities_y,
  //   count,
  //   area_width,
  //   area_height
  // );

  CHECK(cudaMalloc((void**)&positions_velocities_GPU, 4 * count * sizeof(float)));


  // Start timestep timer
  timer.start();

  return count;
}

float* SimulatorGPU::Update(float gravity)
{
  // double start = timer.get_elapsed_ns();

  // int blocks = count / CUDA_BLOCK_SIZE;

  // update_kernel<<<CUDA_BLOCK_SIZE, blocks>>>();
  // cudaDeviceSynchronize();

  // // Compute velocities
  // for (unsigned i = 0; i < count; i++)
  // for (unsigned j = 0; j < count; j++)
  // {
  //   if (i == j) continue;

  //   float diff_x = positions_x[j] - positions_x[i];
  //   float diff_y = positions_y[j] - positions_y[i];

  //   float diff_len_sq = diff_x * diff_x + diff_y * diff_y;
  //   float mult = gravity / diff_len_sq;

  //   velocities_x[i] += mult * diff_x;
  //   velocities_y[i] += mult * diff_y;
  // }

  // double timestep = (timer.get_elapsed_ns() - start) * 0.0000001;

  // // Compute positions
  // for (unsigned i = 0; i < count; i++)
  // {
  //   positions_x[i] += velocities_x[i] * timestep;
  //   positions_y[i] += velocities_y[i] * timestep;
  // }

  return positions_velocities;
}
