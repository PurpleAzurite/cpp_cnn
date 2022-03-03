# cnn_probe
This is prototype for a neural network I'm developing for medical diagnosis purposes. It has a live environment for 3D rendering with OpenGL, as well as a neural network component currently training to be a simple XOR gate.
The modular design of the application gives it flexibility to swap out any component to fit the use-case.

## Dependencies
- CMake
- Ninja (Optional. Supply your own build system.)

## Building
```
git clone --recursive https://github.com/PurpleAzurite/cnn_probe.git
mkdir build && cd build
cmake -G Ninja ../
```

![alt text](https://i.imgur.com/2WheNq3.png)


