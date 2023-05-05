#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <float.h>


inline float sum_vec(__m256 avec) {
  float tmp[8];
  __m256 bvec = _mm256_permute2f128_ps(avec,avec,1);
  bvec = _mm256_add_ps(bvec,avec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  _mm256_store_ps(tmp, bvec);
  return tmp[0];
}

inline void dump_avx(__m256 vec) {
  float tmp[8];
  _mm256_store_ps(tmp, vec);
  for (int i = 0; i< 8; i++) {
    printf("%g ", tmp[i]);
  }
  printf("\n");
}

int main() {
  const int N = 20;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  int vec_N = N - N % 8;
  __m256 zeros = _mm256_setzero_ps();
  __m256 mins = _mm256_set1_ps(-FLT_MAX);
  for(int i=0; i<N; i++) {
    // N は 8 の倍数であると仮定する
    int j;
    for (j=0; j<vec_N; j+=8) {

      // rx, ry の計算
      __m256 rx_vec = _mm256_sub_ps(
        _mm256_set1_ps(x[i]),
        _mm256_load_ps(x+j)
      );
      __m256 ry_vec = _mm256_sub_ps(
        _mm256_set1_ps(y[i]),
        _mm256_load_ps(y+j)
      );
      // r の逆数の計算
      __m256 rr_vec = _mm256_rsqrt_ps(
        _mm256_add_ps(
          _mm256_mul_ps(rx_vec, rx_vec),
          _mm256_mul_ps(ry_vec, ry_vec)
        )
      );
      // mの読み込み
      __m256 m_vec = _mm256_load_ps(m+j);
      // fx, fy それぞれの減る分を計算
      __m256 delta_fx_vec = _mm256_mul_ps(
        _mm256_mul_ps(rx_vec, m_vec),
        _mm256_mul_ps(rr_vec, _mm256_mul_ps(rr_vec, rr_vec))
      );
      __m256 delta_fy_vec = _mm256_mul_ps(
        _mm256_mul_ps(ry_vec, m_vec),
        _mm256_mul_ps(rr_vec, _mm256_mul_ps(rr_vec, rr_vec))
      );
      // maskの作成
      __m256 v_cmp = _mm256_max_ps(delta_fx_vec, mins);  // -nan の削除, -nanの箇所を除外する
      __m256 mask = _mm256_cmp_ps(v_cmp, mins, _CMP_EQ_UQ);
      // maskのbitが立っているところをzeroにする(fx,fy)
      delta_fx_vec = _mm256_blendv_ps(delta_fx_vec, zeros, mask);
      delta_fy_vec = _mm256_blendv_ps(delta_fy_vec, zeros, mask);
      // 各レーンの和を計算
      float sum_fx = sum_vec(delta_fx_vec); 
      float sum_fy = sum_vec(delta_fy_vec); 
      // fx[i], fy[i]に代入
      fx[i] -= sum_fx;
      fy[i] -= sum_fy;
    }
    for (; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
  printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
