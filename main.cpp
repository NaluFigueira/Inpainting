#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

class ConvCovrResult {       
  public:             
    mat t;        
    mat ind;  
    int reg;  
};

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

mat zeropad(mat &m, int M, int N) {
  mat result = zeros(M,N);
  for(int i = 0; i < M/2; i++) {
    for(int j = 0; j < N/2; j++) {
      result(i,j) = m(i,j);
    }
  }
  return result;
}

ConvCovrResult convcovr(mat &t, mat &ind, int reg){
  ConvCovrResult result;
  result.t = t;
  result.ind = ind;
  result.reg = reg;
  return result;
}

mat asdn_periodic(mat &s, mat &mu) {
  int M = mu.n_rows, N = mu.n_cols;
  mat out = zeros(M,N);
  mat W;
  W.randn(M,N);
  cx_mat fW = fft2(W);
  cx_mat out_fft = ifft2(fft2(s)%fW);

  mat result = zeros(M,N);
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      result(i,j) = mu(i,j) + out_fft(i,j).real();
    }
  }

  return result;
}

cx_mat mtimes(ConvCovrResult &A, mat &u) {
  int M = u.n_rows, N = u.n_cols;
  cx_mat ft = fft2(A.t);
  cx_mat conj_ft = conj(ft) % fft2(u); 
  cx_mat tmp = zeros(M, N) + conj_ft;
  cx_mat v = ifft2(ft%tmp);
  v = v + A.reg*A.reg*u;
  for(int i = 0; i < A.ind.n_rows; i++){
    for(int j = 0; j < A.ind.n_cols; j++) {
      if(A.ind(i,j) == 0)
        v(i,j) = 0;
    }
  }

  return v;
}

cx_mat mtimes_complex(ConvCovrResult &A, cx_mat &u) {
  int M = u.n_rows, N = u.n_cols;
  cx_mat ft = fft2(A.t);
  cx_mat conj_ft = conj(ft) % fft2(u); 
  cx_mat tmp = zeros(M, N) + conj_ft;
  cx_mat v = ifft2(ft%tmp);
  v = v + A.reg*A.reg*u;
  for(int i = 0; i < A.ind.n_rows; i++){
    for(int j = 0; j < A.ind.n_cols; j++) {
      if(A.ind(i,j) == 0)
        v(i,j) = 0;
    }
  }

  return v;
}

mat solvecgdnormal(int M, int N, ConvCovrResult A, mat &rhs, int imax, double ep) {
  time_t start, end;
  time(&start); 
  int index = 0;
  mat x = zeros(M, N);
  cx_mat r = mtimes(A, rhs);

  double nr2 = 0;
  double rninf = 0;

  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      nr2 += r(i,j).real()*r(i,j).real();
      double absValue = abs(r(i,j).real());
      if(absValue > rninf) 
        rninf = absValue;
    }
  }

  cx_mat p = r;

  double rn2 = sqrt(nr2); 
  bool stop = false;

  while(index < imax && rn2 > ep && stop == false) {
    index ++;
    cx_mat mtimes_result = mtimes_complex(A, p);
    cx_mat AtAp = mtimes_complex(A, mtimes_result);
    double spAtAp = 0;

    for(int i = 0; i < M; i++) {
      for(int j = 0; j < N; j++) {
        spAtAp += p(i,j).real() * AtAp(i,j).real();
      }
    }  

    double alpha = nr2/spAtAp;

    for(int i = 0; i < M; i++) {
      for(int j = 0; j < N; j++) {
        x(i,j) =  x(i,j) + alpha*p(i,j).real();
        r(i,j) =  r(i,j).real() - alpha*AtAp(i,j).real();
      }
    }
    double nrold2 = nr2;

    nr2 = 0;
    rninf = 0;

    for(int i = 0; i < M; i++) {
      for(int j = 0; j < N; j++) {
        nr2 += r(i,j).real()*r(i,j).real();
        double absValue = abs(r(i,j).real());
        if(absValue > rninf) 
          rninf = absValue;
      }
    }

    rn2 = sqrt(nr2);
    double beta = nr2/nrold2;

    for(int i = 0; i < M; i++) {
      for(int j = 0; j < N; j++) {
        p(i,j) = r(i,j).real() + beta*p(i,j).real();
      }
    }

  }

  time(&end);
  double time_taken = double(end - start); 
  cout << "Time taken by program is : " << fixed 
       << time_taken; 
  cout << " sec " << endl; 

  return x;
}

mat gaussian_inpainting(int M, int N, mat &texture_image, mat &tc, mat &mask, mat &indc, double mean) {
  int imax = 10, reg = 0;
  double ep = 0.000000000000001;
  M = 2*M;
  N = 2*N;
  mat new_texture_image = zeropad(texture_image, M, N),
      new_tc = zeropad(tc, M, N),
      indm = zeropad(mask, M, N),
      new_indc = zeropad(indc, M, N);


  mat mv = zeros(M, N);

  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      mv(i,j) = mean;
    }
  }

  mat ones_mat = ones(M,N);

  ConvCovrResult G = convcovr(new_tc, ones_mat, reg);
  ConvCovrResult A = convcovr(new_tc, new_indc, reg);

  mat zeros_mat = zeros(M,N);

  mat z = asdn_periodic(new_tc, zeros_mat);
  mat rhs = zeros(M,N);

  for(int i = 0; i < new_indc.n_rows; i++) {
    for(int j = 0; j < new_indc.n_cols; j++) {
      if(new_indc(i,j)){
        rhs(i,j) = new_texture_image(i,j) - mv(i,j);
      }
    }
  }

  mat k = solvecgdnormal(M, N, A, rhs, imax, ep);
  rhs = zeros(M,N);
  for(int i = 0; i < new_indc.n_rows; i++) {
    for(int j = 0; j < new_indc.n_cols; j++) {
      if(new_indc(i,j)){
        rhs(i,j) = z(i,j);
      }
    }
  }

  mat zk = solvecgdnormal(M, N, A, rhs, imax, ep);

  cx_mat k_times = mtimes(G, k);
  cx_mat zk_times = mtimes(G, zk);

  mat v = texture_image;

  for(int i = 0; i < indm.n_rows; i++) {
    for(int j = 0; j < indm.n_cols; j++) {
      if(indm(i,j)){
        v(i,j) = mv(i,j) + k_times(i,j).real() + z(i,j) - zk_times(i,j).real();
      }
    }
  }

  int M_Half = floor(M/2), N_Half = floor(N/2);


  mat v_result = zeros(M_Half, N_Half);

  for(int i = 0; i < M_Half; i++) {
    for(int j = 0; j < N_Half; j++) {
      v_result(i,j) = ceil(v(i,j))*255;
    }
  }

  return v_result;
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
        mask(i,j) = 1;
        one_minus_mask(i,j) = 0;
      }
      else {
        one_minus_mask(i,j) = 1;
      } 
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

  int n_nonzeros = 0;

  for(int j = 0; j < N; j++) {
    for(int i = 0; i < M; i++) {
      if(maskrgb(i,j) == 0) {
        n_nonzeros++;
      }
    }
  }

  vec input_mask = zeros(n_nonzeros);

  int w = 0;

  for(int j = 0; j < N; j++) {
    for(int i = 0; i < M; i++) {
      if(maskrgb(i,j) == 0) {
        input_mask(w) = texture_image(i,j);
        w++;
        if(w == n_nonzeros) {
          i = M;
          j = N;
        }
      }
    }
  }

  double mask_mean = mean(input_mask);

  vec input_mask_final = zeros(n_nonzeros);

  for(int i = 0; i < n_nonzeros; i++) {
      input_mask_final(i) = input_mask(i) - mask_mean;
  }

  mat new_value = zeros(M,N);

  for(int i = 0; i < n_nonzeros; i++) {
    new_value(i) = k*input_mask_final(i);
  }

  mat tc = zeros(M,N);

  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
       if(maskrgb(i,j) == 0) {
         tc(i, j) = new_value(i,j);
       }
    }
  }
  
  mat result = gaussian_inpainting(M, N, texture_image, tc, mask, indc, mask_mean);
 
  result.save("output_image.pgm", pgm_binary);

  return 0;
}