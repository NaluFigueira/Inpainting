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
  mat mask, input;

  input.load("input.pgm", pgm_binary);
  mask.load("mask.pgm", pgm_binary);

  mat indc = get_conditioning_points(mask);

  int M = input.n_rows, N = input.n_cols;
  mat one_minus_mask = zeros(M,N);
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      input(i,j) = round(input(i,j)/255);
      if(mask(i,j) == 255) {
        one_minus_mask(i,j) = 0;
      }
      else one_minus_mask(i,j) = 1;
    }
  }

  mat texture_image = input%one_minus_mask;
  mat m = zeros(1,1);
  mat abs_mask = abs(mask);
  mat maskrgb = zeros(M, N);

  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      if(abs_mask(i,j) > 0)
        maskrgb(i,j) = round(abs_mask(i,j)/255);
    }
  }

  sp_mat not_maskrgb = zeros<sp_mat>(M, N);

  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      if(maskrgb(i,j) == 0)
        not_maskrgb(i,j) = 1;
      else
        not_maskrgb(i,j) = 0;
    }
  }

  double k = sqrt(1.0/not_maskrgb.n_nonzero);

  vec input_mask = zeros(M*N);

  int w = 0;

  for(int j = 0; j < N; j++) {
    for(int i = 0; i < M; i++) {
      if(maskrgb(i,j) == 0) {
        input_mask(w) = texture_image(i,j);
        w++;
      }
    }
  }


  double mask_mean = mean(input_mask);

  vec input_mask_final = zeros(M*N);

  for(int i = 0; i < M*N; i++) {
      input_mask_final(i) = input_mask(i) - mask_mean;
  }


  mat new_value = zeros(M,N);

  for(int i = 0; i < N*M; i++) {
    new_value(i) = k*input_mask_final(i);
  }

  mat tc = zeros(M,N);

  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
       if(maskrgb(i,j) == 0) {
         tc(i, j) = new_value(i,j);
         cout << "tc(" << i << "," << j << ")=" << tc(i,j) << "\n";
       }
    }
  }


  return 0;
}