## -*- texinfo -*-
##
## @deftypefn{Function File} {@var{bool} =} isgpuarray (@var{x})
##
## Check if array @var{x} is a of a GPU array
## data type.
##
## @seealso{oclArray, ocl_to_octave, gather}
## @end deftypefn

function flag = isgpuarray (A)
  t = type("A");
  flag = index(t{1}, "OCL array") != 0;
endfunction

%!assert (isgpuarray(gpuArray (1)));
%!assert (! isgpuarray([1 2 3]));
