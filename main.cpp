#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <stdint.h>
#include <GL/freeglut.h>
#include <iostream>

#include "vec2f.hpp"

#include "simulator.hpp"
#include "simulator_cpu.hpp"

const int windowW = 1280;
const int windowH = 720;

const int particles = 1200;
const float gravity = 0.00000001f;

// FPS limiting
const bool limit_refresh = false;
auto last_refresh = std::chrono::system_clock::now();
auto last_stats = std::chrono::system_clock::now();
float refresh_rate = 60;
std::chrono::microseconds refresh_delay_us((int)(1000000.f / refresh_rate));

vec2f* positions = nullptr;

std::chrono::milliseconds stats_delay_ms(1000);
unsigned iterations = 0u;

struct Particle
{
  vec2f position;
  vec2f velocity;
};

Simulator* sim = nullptr;

void init_simulation()
{
  // Init simulator
  sim = new SimulatorCPU();
  sim->Init(particles, windowW, windowH);
}

void wait_for_refresh()
{
  // Check time
  auto since_last_refresh = std::chrono::system_clock::now() - last_refresh;

  // Wait till next refresh
  if (since_last_refresh < refresh_delay_us)
  {
    std::this_thread::sleep_for(refresh_delay_us - since_last_refresh);
    last_refresh = std::chrono::system_clock::now();
  } 
}

void do_stats()
{
  // Check time
  auto since_last_stats = std::chrono::system_clock::now() - last_stats;

  // Show some stats if delay has been reached
  if (since_last_stats > stats_delay_ms)
  {
    unsigned fps = iterations / (stats_delay_ms.count() / 1000);
    unsigned ops = particles * particles * fps;

    std::cout << "fps: " << fps << " - ops/s: " << ops << std::endl;

    iterations = 0u;
    last_stats = std::chrono::system_clock::now();
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

void display()
{
  glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 0.0, 0.0);

	glBegin(GL_POINTS);

  if (positions == nullptr) return;

  for (unsigned i = 0; i < particles; i++)
  {
    glVertex2f(positions[i].x, positions[i].y);
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

  init_simulation();

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  //glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

  glutInitWindowSize(windowW, windowH);
  glutCreateWindow("particles");

  glutDisplayFunc(display);
  glutIdleFunc(update);

  init_display();
  glutMainLoop();

  return 0;
}
