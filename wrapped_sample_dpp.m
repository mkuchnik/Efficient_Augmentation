function Y = wrapped_sample_dpp(M,k)
% decomposes the kernel and samples from it
  debug_on_warning(1);
  dec_L = decompose_kernel(M);
  Y = sample_dpp(dec_L, k);