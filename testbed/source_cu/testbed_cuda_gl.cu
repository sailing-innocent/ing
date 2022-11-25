

// opengl interoperaily
// a buffer object is registered using cudaGraphicsGLRegisterBuffer() -> a device pointer that can be read and written using cudaMemcpy calls
// A texture or renderbuffer object is registered using cudaGraphicsGLRegisterImage() -> CUDA array
// Kernels can read from the array by binding it to a texture or surface reference
// they can also write to it via the surface write functions
// cudaGraphicsRegisterFlagsSurfaceLoadStore()
// cudaMemcpy2D() calls
// internal types of GL_RGBA_FLOAT32, GL_RGBA_8, GL_INTENSITY16 or GL_RGBA8UI
