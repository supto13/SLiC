#ifndef WAVELET_H
#define WAVELET_H

#include <array>
#include <vector>
#include <stdexcept>

#include "vector.h"

//=============================================================================
// Decomposition1D and Decomposition1D2D
// Contain and manage the result of 1D or 2D wavelet decomposition and provide
// accessors to set/get a specific subband.
//=============================================================================

template<typename T>
class Decomposition1D : public std::vector<std::vector<T>> {
public:
  explicit Decomposition1D(size_t num_levels);
  ~Decomposition1D();

  size_t NumLevels() const;

  const std::vector<T>& GetAppcoef() const;
  const std::vector<T>& GetDetcoef(size_t level) const;

  void SetDetcoef(const std::vector<T>& d, size_t level);
  void SetAppcoef(const std::vector<T>& a);
};


//=============================================================================
// Wavelet
// Implementation of 1D and 2D single-level and multi-level Wavelet transform.
//=============================================================================

template<typename T>
class Wavelet {
public:

  // Constructor requires the decomposition/reconstruction filters
  explicit Wavelet(const std::vector<T>& Lo_D, const std::vector<T>& Hi_D,
          const std::vector<T>& Lo_R, const std::vector<T>& Hi_R);
  ~Wavelet();

public:

  //===========================================================================
  // 1D Discrete Wavelet Transform (DWT)
  //===========================================================================

  // Multi-level wavelet decomposition of 1D input vector x
  Decomposition1D<T> Wavedec(const std::vector<T>& x, size_t num_levels) const;

  // Multi-level wavelet reconstruction of 1D vector having length_rec
  std::vector<T> Waverec(const Decomposition1D<T>& decomposition, size_t length_rec) const;

  // Single-level DWT transform of 1D vector x. The output vectors contain the
  // approximation coefficients a and the detail coefficients d
  void Dwt(const std::vector<T>& x, std::vector<T>& a, std::vector<T>& d) const;

  // Single-level inverse DWT transform which computes a 1D vector of length l
  // from the given approximation and detail coefficients
  std::vector<T> Idwt(const std::vector<T>& a, const std::vector<T>& d, size_t l) const;

  //===========================================================================
  // Utils
  //===========================================================================

  // Linearly reconstruct 1D vector assuming the given input a represents the
  // approximation coefficients of a multi-level wavelet decomposition with
  // num_levels. Thus, if vector a has length l, the output will have length
  // equal to l*(2^num_levels)
  std::vector<T> Linrec(const std::vector<T>& a, size_t num_levels) const;

private:

  //===========================================================================
  // Utils for 1D DWT
  //===========================================================================

  // Reconstruct 1D approximation of a vector at given level n from the input
  // decomposition. Using level=0 means full reconstruction, level = 1 means
  // reconstruction at the second-most coarse level, and so on. The length of
  // the reconstructed vector is inferred by the input Decomposition1D and it
  // is equal to the length of the detail coefficients at the next higher
  // level. If level=0 (and thus there is no higher level), the length will be
  // defined as double the length of level zero.
  std::vector<T> Appcoef(const Decomposition1D<T>& decomposition, size_t level) const;

  // Reconstruct the vector approximation at given level and specify the length
  // of the reconstruction using the third argument
  std::vector<T> Appcoef(const Decomposition1D<T>& decomposition, size_t level, size_t length_rec) const;

  // Convolution and decimation
  std::vector<T> Convdown(const std::vector<T>& x, const std::vector<T>& f) const;

  // Upsampling with zero-padding, dyadup, and convolution
  std::vector<T> Upconv(const std::vector<T>& x, const std::vector<T>& f, size_t s) const;

  //===========================================================================
  // Utils
  //===========================================================================

  // Double the size of x interleaving zeros as  {x1, 0, x2, 0, x3, ... xn, 0}
  std::vector<T> Dyadup(const std::vector<T>& x) const;

  // Periodized extension of x
  std::vector<T> Wextend(const std::vector<T>& x, size_t lenEXT) const;

private:

  //===========================================================================
  // Member variables
  //===========================================================================

  std::vector<T> Lo_D_; // Decomposition low-pass filter
  std::vector<T> Hi_D_; // Decomposition high-pass filter
  std::vector<T> Lo_R_; // Reconstruction low-pass filter
  std::vector<T> Hi_R_; // Reconstruction high-pass filter

  size_t half_length_filter_;

}; // class Wavelet<T>

#endif // WAVELET_H
