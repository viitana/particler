#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <stdint.h>

#include <iostream>

#include "../packages/plf_nanotimer/plf_nanotimer.hpp"

#include "simulator.hpp"
#include "simulator_single_cpu.hpp"
#include "simulator_multi_cpu.hpp"
#include "simulator_single_simd.hpp"
#include "simulator_multi_simd.hpp"
#include "simulator_gpu.cuh"

#include "GL/glut.h"

#include <immintrin.h>
#include <limits>

const int windowW = 1920;
const int windowH = 1080;

int particles = 40960; //5120;
const float gravity = 0.01f;

const bool limit_refresh = false;

const double refresh_rate = 200;
const double refresh_delay_us = 1000000. / refresh_rate;
const double stats_rate =  1;
const double stats_delay_us = 1000000. / stats_rate;

double last_refresh_us = 0;
double last_stats_us = 0;
plf::nanotimer timer;

float* positions = nullptr;

unsigned iterations = 0u;

Simulator* sim = nullptr;

void init_simulation()
{
  // Init simulator
  //sim = new SimulatorSingleCPU();
  //sim = new SimulatorMultiCPU();
  //sim = new SimulatorSingleSIMD();
  //sim = new SimulatorMultiSIMD();
  sim = new SimulatorGPU();

  particles = sim->Init(particles, windowW, windowH);
}

void wait_for_refresh()
{
  // Check time
  double since_last_refresh_us = timer.get_elapsed_us() - last_refresh_us;

  // Wait till next refresh
  plf::microsecond_delay(refresh_delay_us - since_last_refresh_us);
  last_refresh_us = timer.get_elapsed_us();
}

void do_stats()
{
  // Check time
  auto since_last_stats_us = timer.get_elapsed_us() - last_stats_us;

  // Show some stats if delay has been reached
  if (since_last_stats_us > stats_delay_us)
  {
    unsigned fps = iterations / (stats_delay_us / 1000000);
    unsigned long long particles_long = particles;
    unsigned long long interactions = particles_long * particles_long * fps;

    std::cout << "fps: " << fps << " - interactions/s: " << interactions << std::endl;

    iterations = 0u;
    last_stats_us = timer.get_elapsed_us();
  }
}

void update()
{
  // Do physics
  positions = sim->Update(gravity);
  iterations++;

  if (limit_refresh)
  {
    wait_for_refresh();
  }

  do_stats();

  // Refresh
  glutPostRedisplay();
}

void render()
{
  glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 0.0, 0.0);

	glBegin(GL_POINTS);

  if (positions == nullptr) return;

  for (unsigned i = 0; i < particles; i++)
  {
    glVertex2f(
      positions[i],
      positions[particles + i]
    );
  }

	glEnd();
	glFlush();
}

void init_display()
{
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glColor3f(1.0, 0.0, 0.0);
  glPointSize(3.0);

  glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, windowW, 0.0, windowH);
}

int main(int argc, char **argv)
{
  std::cout << "Starting particles" << std::endl;

  timer.start();

  init_simulation();

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  //glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

  glutInitWindowSize(windowW, windowH);
  glutCreateWindow("particles");

  glutDisplayFunc(render);
  glutIdleFunc(update);

  init_display();
  glutMainLoop();

  return 0;
}
