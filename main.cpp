#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

mat get_conditioning_points(mat &mask) {
  int M = mask.n_rows, N = mask.n_cols;
  int M2 = 2*M, N2 = 2*N;
  
  mat bmask = zeros(M2, N2);
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      if(mask(i,j) > 0)
        bmask(i,j) = round(mask(i,j)/255);
    }
  }


  mat k = zeros(M2, N2), k_shift = zeros(M2, N2);
  k(span(0,6), span(0,6)).ones();

  //---------------------------------------
  //circshift
  for (int i =0; i < M2; i++) {
    int ii = (i - 3) % M2;
    if (ii<0) ii = M2 + ii;
    for (int j = 0; j < N2; j++) {
      int jj = (j - 3) % N2;
      if (jj<0) jj = N2 + jj;
      k_shift(ii, jj) = k(i, j);
    }
  }
  //--------------------------------------------

  cx_mat fft2_bmask = fft2(bmask);
  cx_mat fft2_k = fft2(k_shift);
  

  cx_mat co = ifft2(fft2_bmask%fft2_k);
  mat co_aux = zeros(M2, N2);

  for (int i = 0; i < M2; i++) {
    for(int j = 0; j < N2; j++) {
      if(co(i,j).real() > 0.5) {
        co_aux(i,j) = 1;
      }
    }
  }
  
  for(int i = 0; i < M2; i++) {
    for(int j = 0; j < N2; j++) {
      if(bmask(i,j) > 0) {
        co_aux(i,j) = 0;
      }
    }
  }

  mat co_normal_size = zeros(M,N);
 
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      co_normal_size(i,j) = co_aux(i,j);
    }
  }

  mat result = repmat(co_normal_size, 1, 1);

  return result;
}

int main(int argc, char** argv) {
  mat mask;

  mask.load("mask.pgm", pgm_binary);

  mat result = get_conditioning_points(mask);

    

  result.save("output_image.pgm", pgm_binary);

  return 0;
}