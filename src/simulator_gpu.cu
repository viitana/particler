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

__global__ void velocities(float* positions_velocities, int count, float gravity)
{
  int i = CUDA_BLOCK_SIZE * blockIdx.x + threadIdx.x;

  float* positions_x = positions_velocities;
  float* positions_y = positions_velocities + (1 * count);
  float* velocities_x = positions_velocities + (2 * count);
  float* velocities_y = positions_velocities + (3 * count);

  for (unsigned j = 0; j < count; j++)
  {
    if (i == j) continue;

    float diff_x = positions_x[j] - positions_x[i];
    float diff_y = positions_y[j] - positions_y[i];

    float diff_len_sq = diff_x * diff_x + diff_y * diff_y;
    float mult = gravity / diff_len_sq;

    velocities_x[i] += mult * diff_x;
    velocities_y[i] += mult * diff_y;
    }
}

__global__ void positions(float* positions_velocities, int count, double timestep)
{
  int i = CUDA_BLOCK_SIZE * blockIdx.x + threadIdx.x;

  float* positions_x = positions_velocities;
  float* positions_y = positions_velocities + (1 * count);
  float* velocities_x = positions_velocities + (2 * count);
  float* velocities_y = positions_velocities + (3 * count);

  positions_x[i] += velocities_x[i] * timestep;
  positions_y[i] += velocities_y[i] * timestep;
}


int SimulatorGPU::Init(int particles, int area_width, int area_height)
{
  count = util::round_up(particles, CUDA_BLOCK_SIZE);

  // Init particles
  srand(2277);

  positions_velocities = new float[4 * count];

  util::init_particles_circle(
    positions_velocities,
    positions_velocities + 1 * count,
    positions_velocities + 2 * count,
    positions_velocities + 3 * count,
    count,
    area_width,
    area_height
  );

  // Allocate GPU memory
  CHECK(cudaMalloc((void**)&positions_velocities_GPU, 4 * count * sizeof(float)));

  // Copy data to GPU
  CHECK(cudaMemcpy(positions_velocities_GPU, positions_velocities, 4 * count * sizeof(float), cudaMemcpyHostToDevice));

  // Start timestep timer
  timer.start();

  return count;
}

float* SimulatorGPU::Update(float gravity)
{
  double start = timer.get_elapsed_ns();

  int blocks = count / CUDA_BLOCK_SIZE;

  float* positions_x = positions_velocities;
  float* positions_y = positions_velocities + 1 * count;
  float* velocities_x = positions_velocities + 2 * count;
  float* velocities_y = positions_velocities + 3 * count;

  velocities<<<blocks, CUDA_BLOCK_SIZE>>>(positions_velocities_GPU, count, gravity);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  double timestep = (timer.get_elapsed_ns() - start) * 0.0000001;

  positions<<<blocks, CUDA_BLOCK_SIZE>>>(positions_velocities_GPU, count, timestep);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(positions_velocities, positions_velocities_GPU, 2 * count * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  return positions_velocities;
}
