# particler

A simple brute-force particle simulator intended for testing various parallel programming approaches

### Prerequisites

- OpenGL
- CUDA toolkit (tested on 9.1 and higher)
- cmake (3.18 or higher)

### Building

Building has been tested on Windows 10 x64 and Ubuntu 18.04 x64.

From the repo root directory, switch to some lower directory you want to build in, for example `build` and run
```
cmake ..
```
To generate a build system approppriate for your platform. 

#### Linux
On Linux, cmake will likely generate a GNU makefile. You can then build with
```
make
```
#### Windows
On Linux, cmake will likely generate a Visual Studio solution. You can then build with
```
msbuild particles.sln
```

### Troubleshooting

#### Windows

If you get errors related to no compiler being found or `msbuild` is not recogniced, make sure you are using a [command line that has the Microsoft C++ toolset enabled](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line).
