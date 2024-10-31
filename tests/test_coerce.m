pkg load ocl

a = gpuArray([ 1 2; 3 4 ]);
b = [5 6 ; 7 8];

assert (a .* b == b .* a);
assert(min(a,b) == min(b,a))
