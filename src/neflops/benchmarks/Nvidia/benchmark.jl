A32 = CUDA.rand(Float32,3,3);
B32 = CUDA.rand(Float32,3,3);
A64 = CUDA.rand(Float64,3,3);
B64 = CUDA.rand(Float64,3,3);

A32 * B32;
B32 * A32;
B32 * B32;
A32 * A32;

A64 * B64;
B64 * A64;
B64 * B64;
A64 * A64;

sizes = [30,90,100,1024,10000]
# Single precision
CUDA.@profile for sz in sizes
    A = CUDA.rand(Float32,sz,sz);
    B = CUDA.rand(Float32,sz,sz);
    A * B;
    B * A;
    A * A;
    B * B;
end

# Double precision
CUDA.@profile for sz in sizes
    A = CUDA.rand(Float64,sz,sz);
    B = CUDA.rand(Float64,sz,sz);
    A * B;
    B * A;
    A * A;
    B * B;
end