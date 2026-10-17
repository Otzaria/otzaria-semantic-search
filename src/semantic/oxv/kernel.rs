//! The dot product a scan runs once per stored vector, in exact integer arithmetic.
//!
//! An `i8-sym-dim` component stands for `v_d · S_d / 127`, so a query `q` scores a vector as
//! `Σ q_d · S_d / 127 · v_d`. [`PreparedQuery`] folds the scales into the query once —
//! `w_d = q_d · S_d / 127` — and turns it into 16-bit integers, `q16 = round(w · t)`, with
//! the largest `t` for which no sum can overflow:
//!
//! ```text
//! t = min(32767 / max|w|, (2³¹ − 1 − 127 · dim / 2) / (127 · Σ|w|))
//! ```
//!
//! The first bound keeps every `q16` inside `i16`; the second keeps `Σ |q16 · v|` — at most
//! `127 · Σ|q16|`, rounding included — inside `i32`. A score is then `acc · (1 / t)`.
//!
//! An `i8-sym-vec` component stands for `v_d · s`, with `s` the vector's own scale, so a
//! query scores it as `s · Σ q_d · v_d`: the query is quantized as it is — `w = q` — under
//! the same two bounds, which hold whatever `s` is, and the score is `s · acc · (1 / t)`.
//! The scale is applied in `f32` after the integer sum, in that order on every CPU, so the
//! score is the same number everywhere.
//!
//! A `rot192-i8i4` vector is `ŷ = (s · σ_h · c_h, lo_t + q_t · step_t)` in the rotated space,
//! so a query `q` is rotated once, `r = R · q`, and scores it as
//!
//! ```text
//! score = s · Σ_h (r_h · σ_h) · c_h  +  Σ_t r_t · lo_t  +  Σ_t (r_t · step_t) · q_t
//!       = (acc8 · s) · (1 / t8)  +  (C + acc4 · (1 / t4))
//! ```
//!
//! — two exact integer sums, `acc8` of 16-bit head weights against the int8 codes and `acc4`
//! of 16-bit tail weights against the 4-bit codes, each quantized under its own bound with
//! the codes' largest magnitude (127 and 15), and a constant `C` of the query alone. The
//! floats are combined in that order, so the score is again the same number on every CPU,
//! and it is the cosine of the query with `Rᵀ · ŷ`, the vector decoded.
//!
//! **Every kernel computes the same integer.** Integer addition is associative, so the
//! scalar loop, AVX2's `madd` and NEON's `mlal` reach the same `acc` whatever order they add
//! in, on every CPU — and wrapping, should the bound ever be wrong, wraps them all alike.
//! That is what makes a scan's ranking, and its goldens, portable. The tests hold each SIMD
//! kernel to the scalar one on random and extreme inputs.
//!
//! The `f32` codec, the reference, has a kernel too: eight lanes summed in a fixed order,
//! which IEEE arithmetic evaluates identically everywhere, since Rust never contracts or
//! reorders a float expression on its own.

use crate::errors::VectorStoreError;
use crate::semantic::oxv::codec::Codec;

/// A kernel: the int8 components as the segment stores them, against a prepared query.
pub(crate) type DotKernel = fn(&[i16], &[u8]) -> i32;

/// A 4-bit kernel: the low nibble of byte `j` against `lo[j]`, the high one against `hi[j]`.
pub(crate) type NibbleKernel = fn(&[i16], &[i16], &[u8]) -> i32;

/// A query, prepared once for the codec epoch of the set it scans.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PreparedQuery {
    /// For `i8-sym-dim`: the scales folded in.
    Int8 {
        q16: Box<[i16]>,
        inv_scale: f32,
    },
    /// For `i8-sym-vec`: each vector's scale applied to its sum.
    Int8PerVector {
        q16: Box<[i16]>,
        inv_scale: f32,
    },
    /// For `rot192-i8i4`: the query rotated, its head and tail quantized apart.
    Rotated {
        /// The query's width, before the rotation.
        dim: usize,
        head: Box<[i16]>,
        inv_head: f32,
        /// The tail's weights for the even components — the low nibbles — and the odd ones.
        tail_lo: Box<[i16]>,
        tail_hi: Box<[i16]>,
        inv_tail: f32,
        /// `Σ r_t · lo_t`: what every vector's tail is worth before its codes.
        constant: f32,
    },
    Float {
        q: Box<[f32]>,
    },
}

impl PreparedQuery {
    /// Fold `codec`'s scales into `query` and quantize it, or keep it as floats for `f32`.
    pub(crate) fn new(query: &[f32], codec: &Codec) -> Result<Self, VectorStoreError> {
        if query.len() != codec.dim() {
            return Err(VectorStoreError::DimensionMismatch {
                store_dim: codec.dim() as u32,
                vector_dim: query.len() as u32,
            });
        }
        if let Some(rotated) = codec.rotated() {
            let mut r = vec![0f64; rotated.out_dim()];
            crate::semantic::oxv::rotation::rotate(&rotated.rotation, query, &mut r);
            let head = r[..rotated.head]
                .iter()
                .zip(rotated.scale.iter())
                .map(|(r, spread)| r * f64::from(*spread))
                .collect();
            let (head, inv_head) = quantize_query(head, 127);
            let tail = &r[rotated.head..];
            let weights = tail
                .iter()
                .enumerate()
                .map(|(d, r)| r * f64::from(rotated.step(d)))
                .collect();
            let mut constant = 0f64;
            for (r, lo) in tail.iter().zip(rotated.lo.iter()) {
                constant += r * f64::from(*lo);
            }
            let (tail, inv_tail) = quantize_query(weights, 15);
            return Ok(Self::Rotated {
                dim: query.len(),
                head,
                inv_head,
                tail_lo: tail.iter().step_by(2).copied().collect(),
                tail_hi: tail.iter().skip(1).step_by(2).copied().collect(),
                inv_tail,
                constant: constant as f32,
            });
        }
        Ok(match codec.scales() {
            Some(scales) => {
                let weights = query
                    .iter()
                    .zip(scales)
                    .map(|(q, s)| f64::from(*q) * f64::from(*s) / 127.0)
                    .collect();
                let (q16, inv_scale) = quantize_query(weights, 127);
                Self::Int8 { q16, inv_scale }
            }
            None if codec.has_vector_scales() => {
                let (q16, inv_scale) =
                    quantize_query(query.iter().map(|q| f64::from(*q)).collect(), 127);
                Self::Int8PerVector { q16, inv_scale }
            }
            None => Self::Float { q: query.into() },
        })
    }

    /// The width of the query.
    pub(crate) fn dim(&self) -> usize {
        match self {
            Self::Int8 { q16, .. } | Self::Int8PerVector { q16, .. } => q16.len(),
            Self::Rotated { dim, .. } => *dim,
            Self::Float { q } => q.len(),
        }
    }
}

/// `q16` and `1 / t` for the weights `w` against codes of magnitude at most `code_max`, per
/// the module documentation. In `f64`, then checked in integers: the bound is the guarantee,
/// so it is verified rather than trusted to the rounding of `t`.
fn quantize_query(w: Vec<f64>, code_max: u32) -> (Box<[i16]>, f32) {
    let code = f64::from(code_max);
    let max = w.iter().fold(0.0f64, |max, x| max.max(x.abs()));
    let sum: f64 = w.iter().map(|x| x.abs()).sum();
    if !(max > 0.0 && max.is_finite() && sum.is_finite()) {
        // A zero query scores everything zero; the runtime never hands one over.
        return (vec![0; w.len()].into_boxed_slice(), 0.0);
    }
    let headroom = f64::from(i32::MAX) - code * w.len() as f64 / 2.0;
    let mut t = (32767.0 / max).min(headroom / (code * sum));
    loop {
        let q16: Vec<i16> = w
            .iter()
            .map(|x| (x * t).round().clamp(-32767.0, 32767.0) as i16)
            .collect();
        let bound: i64 = q16
            .iter()
            .map(|q| i64::from(q.unsigned_abs()) * i64::from(code_max))
            .sum();
        if bound <= i64::from(i32::MAX) {
            return (q16.into_boxed_slice(), (1.0 / t) as f32);
        }
        t *= 0.999_999;
    }
}

/// The reference kernel: sixteen lanes, which the compiler vectorizes on its own.
pub(crate) fn dot_scalar(q: &[i16], v: &[u8]) -> i32 {
    let n = q.len().min(v.len());
    let (q_chunks, q_rest) = q[..n].as_chunks::<16>();
    let (v_chunks, v_rest) = v[..n].as_chunks::<16>();
    let mut lanes = [0i32; 16];
    for (q, v) in q_chunks.iter().zip(v_chunks) {
        for lane in 0..16 {
            lanes[lane] = lanes[lane].wrapping_add(i32::from(q[lane]) * i32::from(v[lane] as i8));
        }
    }
    let mut total = lanes.iter().fold(0i32, |sum, lane| sum.wrapping_add(*lane));
    for (q, v) in q_rest.iter().zip(v_rest) {
        total = total.wrapping_add(i32::from(*q) * i32::from(*v as i8));
    }
    total
}

/// AVX2: sixteen components a step, widened to 16 bits and multiplied pairwise into eight
/// 32-bit sums by `vpmaddwd`.
///
/// # Safety
///
/// The CPU must support AVX2. [`select`] only hands this out after detecting it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(q: &[i16], v: &[u8]) -> i32 {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
        _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_shuffle_epi32,
    };
    let n = q.len().min(v.len());
    let steps = n / 16;
    let mut acc = _mm256_setzero_si256();
    for step in 0..steps {
        let at = step * 16;
        // SAFETY: `at + 16 <= n`, which both slices are at least as long as, and both loads
        // are unaligned ones.
        let (bytes, query) = unsafe {
            (
                _mm_loadu_si128(v.as_ptr().add(at).cast::<__m128i>()),
                _mm256_loadu_si256(q.as_ptr().add(at).cast::<__m256i>()),
            )
        };
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(query, _mm256_cvtepi8_epi16(bytes)));
    }
    let halves = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let pairs = _mm_add_epi32(halves, _mm_shuffle_epi32::<0b01_00_11_10>(halves));
    let all = _mm_add_epi32(pairs, _mm_shuffle_epi32::<0b10_11_00_01>(pairs));
    let mut total = _mm_cvtsi128_si32(all);
    for at in steps * 16..n {
        total = total.wrapping_add(i32::from(q[at]) * i32::from(v[at] as i8));
    }
    total
}

/// The safe entry [`select`] returns once AVX2 is known to be there.
#[cfg(target_arch = "x86_64")]
fn dot_avx2_detected(q: &[i16], v: &[u8]) -> i32 {
    // SAFETY: only reachable through `select`, which returns it after
    // `is_x86_feature_detected!("avx2")` — and through the tests, which check the same.
    unsafe { dot_avx2(q, v) }
}

/// NEON: sixteen components a step, widened to 16 bits and multiply-accumulated into four
/// 32-bit lanes twice over by `smlal`/`smlal2`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn dot_neon(q: &[i16], v: &[u8]) -> i32 {
    use std::arch::aarch64::{
        vaddq_s32, vaddvq_s32, vdupq_n_s32, vget_low_s16, vget_low_s8, vld1q_s16, vld1q_s8,
        vmlal_high_s16, vmlal_s16, vmovl_high_s8, vmovl_s8,
    };
    let n = q.len().min(v.len());
    let steps = n / 16;
    let (mut low, mut high) = (vdupq_n_s32(0), vdupq_n_s32(0));
    for step in 0..steps {
        let at = step * 16;
        // SAFETY: `at + 16 <= n`, which both slices are at least as long as: 16 bytes of
        // `v` and twice 8 lanes of `q`.
        let (bytes, q0, q1) = unsafe {
            (
                vld1q_s8(v.as_ptr().add(at).cast::<i8>()),
                vld1q_s16(q.as_ptr().add(at)),
                vld1q_s16(q.as_ptr().add(at + 8)),
            )
        };
        let v0 = vmovl_s8(vget_low_s8(bytes));
        let v1 = vmovl_high_s8(bytes);
        low = vmlal_s16(low, vget_low_s16(q0), vget_low_s16(v0));
        high = vmlal_high_s16(high, q0, v0);
        low = vmlal_s16(low, vget_low_s16(q1), vget_low_s16(v1));
        high = vmlal_high_s16(high, q1, v1);
    }
    let mut total = vaddvq_s32(vaddq_s32(low, high));
    for at in steps * 16..n {
        total = total.wrapping_add(i32::from(q[at]) * i32::from(v[at] as i8));
    }
    total
}

/// The entry [`select`] returns on aarch64.
#[cfg(target_arch = "aarch64")]
fn dot_neon_baseline(q: &[i16], v: &[u8]) -> i32 {
    // SAFETY: NEON is part of the aarch64 baseline — every aarch64 CPU, and every aarch64
    // target Rust builds for, has it — so the feature `dot_neon` asks for is always there.
    unsafe { dot_neon(q, v) }
}

/// The fastest kernel this CPU has, and its name. AVX2 is detected at run time; NEON is
/// part of every aarch64 CPU; everything else gets the scalar kernel, which the compiler
/// vectorizes for the baseline it targets (SSE2 on x86-64).
pub(crate) fn select() -> (DotKernel, &'static str) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return (dot_avx2_detected, "avx2");
    }
    #[cfg(target_arch = "aarch64")]
    return (dot_neon_baseline, "neon");
    #[allow(unreachable_code)]
    (dot_scalar, "scalar")
}

/// Every kernel this CPU can run, for the tests that hold them to the scalar one.
#[cfg(test)]
pub(crate) fn available() -> Vec<(DotKernel, &'static str)> {
    let mut kernels: Vec<(DotKernel, &'static str)> = vec![(dot_scalar, "scalar")];
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        kernels.push((dot_avx2_detected, "avx2"));
    }
    #[cfg(target_arch = "aarch64")]
    kernels.push((dot_neon_baseline, "neon"));
    kernels
}

/// The reference 4-bit kernel.
pub(crate) fn nibbles_scalar(lo: &[i16], hi: &[i16], packed: &[u8]) -> i32 {
    let mut total = 0i32;
    for ((lo, hi), byte) in lo.iter().zip(hi).zip(packed) {
        total = total
            .wrapping_add(i32::from(*lo) * i32::from(byte & 0x0F))
            .wrapping_add(i32::from(*hi) * i32::from(byte >> 4));
    }
    total
}

/// AVX2: sixteen bytes a step, widened to 16 bits, split into their nibbles and multiplied
/// pairwise against the even and the odd weights by `vpmaddwd`.
///
/// # Safety
///
/// The CPU must support AVX2. [`select_nibbles`] only hands this out after detecting it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nibbles_avx2(lo: &[i16], hi: &[i16], packed: &[u8]) -> i32 {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi256_si128,
        _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_srli_epi16, _mm_add_epi32,
        _mm_cvtsi128_si32, _mm_loadu_si128, _mm_shuffle_epi32,
    };
    let n = lo.len().min(hi.len()).min(packed.len());
    let steps = n / 16;
    let mask = _mm256_set1_epi16(0x0F);
    let mut acc = _mm256_setzero_si256();
    for step in 0..steps {
        let at = step * 16;
        // SAFETY: `at + 16 <= n`, which all three slices are at least as long as, and every
        // load is an unaligned one.
        let (bytes, low_q, high_q) = unsafe {
            (
                _mm_loadu_si128(packed.as_ptr().add(at).cast::<__m128i>()),
                _mm256_loadu_si256(lo.as_ptr().add(at).cast::<__m256i>()),
                _mm256_loadu_si256(hi.as_ptr().add(at).cast::<__m256i>()),
            )
        };
        let wide = _mm256_cvtepu8_epi16(bytes);
        let low = _mm256_and_si256(wide, mask);
        let high = _mm256_srli_epi16::<4>(wide);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(low_q, low));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(high_q, high));
    }
    let halves = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let pairs = _mm_add_epi32(halves, _mm_shuffle_epi32::<0b01_00_11_10>(halves));
    let all = _mm_add_epi32(pairs, _mm_shuffle_epi32::<0b10_11_00_01>(pairs));
    let total = _mm_cvtsi128_si32(all);
    let rest = steps * 16;
    total.wrapping_add(nibbles_scalar(&lo[rest..n], &hi[rest..n], &packed[rest..n]))
}

/// The safe entry [`select_nibbles`] returns once AVX2 is known to be there.
#[cfg(target_arch = "x86_64")]
fn nibbles_avx2_detected(lo: &[i16], hi: &[i16], packed: &[u8]) -> i32 {
    // SAFETY: only reachable through `select_nibbles`, which returns it after
    // `is_x86_feature_detected!("avx2")` — and through the tests, which check the same.
    unsafe { nibbles_avx2(lo, hi, packed) }
}

/// NEON: sixteen bytes a step, widened to 16 bits, split into their nibbles and
/// multiply-accumulated into four 32-bit lanes by `smlal`/`smlal2`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn nibbles_neon(lo: &[i16], hi: &[i16], packed: &[u8]) -> i32 {
    use std::arch::aarch64::{
        vaddq_s32, vaddvq_s32, vandq_u16, vdupq_n_s32, vdupq_n_u16, vget_low_s16, vget_low_u8,
        vld1q_s16, vld1q_u8, vmlal_high_s16, vmlal_s16, vmovl_high_u8, vmovl_u8,
        vreinterpretq_s16_u16, vshrq_n_u16,
    };
    let n = lo.len().min(hi.len()).min(packed.len());
    let steps = n / 16;
    let mask = vdupq_n_u16(0x0F);
    let (mut low_acc, mut high_acc) = (vdupq_n_s32(0), vdupq_n_s32(0));
    for step in 0..steps {
        let at = step * 16;
        // SAFETY: `at + 16 <= n`, which all three slices are at least as long as.
        let (bytes, lo0, lo1, hi0, hi1) = unsafe {
            (
                vld1q_u8(packed.as_ptr().add(at)),
                vld1q_s16(lo.as_ptr().add(at)),
                vld1q_s16(lo.as_ptr().add(at + 8)),
                vld1q_s16(hi.as_ptr().add(at)),
                vld1q_s16(hi.as_ptr().add(at + 8)),
            )
        };
        let (w0, w1) = (vmovl_u8(vget_low_u8(bytes)), vmovl_high_u8(bytes));
        let (l0, l1) = (
            vreinterpretq_s16_u16(vandq_u16(w0, mask)),
            vreinterpretq_s16_u16(vandq_u16(w1, mask)),
        );
        let (h0, h1) = (
            vreinterpretq_s16_u16(vshrq_n_u16::<4>(w0)),
            vreinterpretq_s16_u16(vshrq_n_u16::<4>(w1)),
        );
        low_acc = vmlal_s16(low_acc, vget_low_s16(lo0), vget_low_s16(l0));
        high_acc = vmlal_high_s16(high_acc, lo0, l0);
        low_acc = vmlal_s16(low_acc, vget_low_s16(lo1), vget_low_s16(l1));
        high_acc = vmlal_high_s16(high_acc, lo1, l1);
        low_acc = vmlal_s16(low_acc, vget_low_s16(hi0), vget_low_s16(h0));
        high_acc = vmlal_high_s16(high_acc, hi0, h0);
        low_acc = vmlal_s16(low_acc, vget_low_s16(hi1), vget_low_s16(h1));
        high_acc = vmlal_high_s16(high_acc, hi1, h1);
    }
    let total = vaddvq_s32(vaddq_s32(low_acc, high_acc));
    let rest = steps * 16;
    total.wrapping_add(nibbles_scalar(&lo[rest..n], &hi[rest..n], &packed[rest..n]))
}

/// The entry [`select_nibbles`] returns on aarch64.
#[cfg(target_arch = "aarch64")]
fn nibbles_neon_baseline(lo: &[i16], hi: &[i16], packed: &[u8]) -> i32 {
    // SAFETY: NEON is part of the aarch64 baseline, as for `dot_neon_baseline`.
    unsafe { nibbles_neon(lo, hi, packed) }
}

/// The fastest 4-bit kernel this CPU has, chosen as [`select`] chooses.
pub(crate) fn select_nibbles() -> (NibbleKernel, &'static str) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return (nibbles_avx2_detected, "avx2");
    }
    #[cfg(target_arch = "aarch64")]
    return (nibbles_neon_baseline, "neon");
    #[allow(unreachable_code)]
    (nibbles_scalar, "scalar")
}

/// Every 4-bit kernel this CPU can run.
#[cfg(test)]
pub(crate) fn available_nibbles() -> Vec<(NibbleKernel, &'static str)> {
    let mut kernels: Vec<(NibbleKernel, &'static str)> = vec![(nibbles_scalar, "scalar")];
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        kernels.push((nibbles_avx2_detected, "avx2"));
    }
    #[cfg(target_arch = "aarch64")]
    kernels.push((nibbles_neon_baseline, "neon"));
    kernels
}

/// The `f32` codec's dot product: eight lanes, summed in one fixed order.
pub(crate) fn dot_f32(q: &[f32], v: &[u8]) -> f32 {
    let n = q.len().min(v.len() / 4);
    let (q_chunks, q_rest) = q[..n].as_chunks::<8>();
    let (v_chunks, _) = v[..n * 4].as_chunks::<32>();
    let mut lanes = [0f32; 8];
    for (q, v) in q_chunks.iter().zip(v_chunks) {
        let (v, _) = v.as_chunks::<4>();
        for lane in 0..8 {
            lanes[lane] += q[lane] * f32::from_le_bytes(v[lane]);
        }
    }
    let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    let tail = &v[q_chunks.len() * 32..n * 4];
    for (q, v) in q_rest.iter().zip(tail.as_chunks::<4>().0) {
        total += q * f32::from_le_bytes(*v);
    }
    total
}

/// An `f32` score as an `i32` that orders as the floats do — the scan ranks every codec by
/// an integer.
pub(crate) fn ordered(score: f32) -> i32 {
    let bits = score.to_bits() as i32;
    bits ^ (((bits >> 31) as u32) >> 1) as i32
}

/// The inverse of [`ordered`].
pub(crate) fn unordered(rank: i32) -> f32 {
    f32::from_bits((rank ^ (((rank >> 31) as u32) >> 1) as i32) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::oxv::testing::Random;

    fn random_case(random: &mut Random, dim: usize, extreme: bool) -> (Vec<i16>, Vec<u8>) {
        let q = (0..dim)
            .map(|_| {
                if extreme {
                    if random.next().is_multiple_of(2) {
                        32767
                    } else {
                        -32767
                    }
                } else {
                    (random.next() % 65535) as i32 as i16
                }
            })
            .collect();
        let v = (0..dim)
            .map(|_| {
                if extreme {
                    (if random.next().is_multiple_of(2) {
                        127i8
                    } else {
                        -127
                    }) as u8
                } else {
                    (random.next() % 255) as u8
                }
            })
            .collect();
        (q, v)
    }

    fn exact(q: &[i16], v: &[u8]) -> i64 {
        q.iter()
            .zip(v)
            .map(|(q, v)| i64::from(*q) * i64::from(*v as i8))
            .sum()
    }

    /// Every kernel this CPU has gives the scalar kernel's integer, bit for bit, at every
    /// width from 1 to 300 — tails included — on random and on extreme inputs.
    #[test]
    fn every_kernel_computes_the_scalar_kernels_integer() {
        let kernels = available();
        assert!(kernels.iter().any(|(_, name)| *name == select().1));
        let mut random = Random(21);
        for dim in 1..=300 {
            for extreme in [false, true] {
                let (q, v) = random_case(&mut random, dim, extreme);
                let reference = dot_scalar(&q, &v);
                // 300 · 32768 · 128 is inside i32, so every case here is exact.
                assert_eq!(i64::from(reference), exact(&q, &v), "dim {dim}");
                for (kernel, name) in &kernels {
                    assert_eq!(kernel(&q, &v), reference, "{name} at dim {dim}");
                }
            }
        }
    }

    /// The scale makes overflow impossible: for queries built to be hard — one dominant
    /// component, all equal, alternating signs — every `q16` fits, `127 · Σ|q16|` fits, and
    /// the worst vector for each scores exactly what 64-bit arithmetic says.
    #[test]
    fn a_prepared_query_cannot_overflow_any_kernel() {
        let dim = 256;
        let mut queries: Vec<Vec<f32>> = vec![
            {
                let mut q = vec![1e-4f32; dim];
                q[7] = 1.0;
                q
            },
            vec![1.0 / (dim as f32).sqrt(); dim],
            (0..dim)
                .map(|i| if i % 2 == 0 { 0.0625 } else { -0.0625 })
                .collect(),
        ];
        let mut random = Random(22);
        queries.extend((0..20).map(|_| random.unit(dim)));
        let scales: Vec<Vec<f32>> = vec![
            vec![1.0; dim],
            (0..dim).map(|i| 0.01 + i as f32 / 300.0).collect(),
        ];

        for scales in &scales {
            let codec = Codec::i8_sym_dim(scales.clone(), 1.0).unwrap();
            for query in &queries {
                let PreparedQuery::Int8 { q16, inv_scale } =
                    PreparedQuery::new(query, &codec).unwrap()
                else {
                    panic!("an int8 codec prepares an int8 query");
                };
                assert!(inv_scale > 0.0);
                let bound: i64 = q16.iter().map(|q| i64::from(q.unsigned_abs()) * 127).sum();
                assert!(bound <= i64::from(i32::MAX), "bound {bound}");
                assert!(
                    q16.iter().any(|q| q.unsigned_abs() > 16000),
                    "t is not wasted"
                );
                // The vector that makes every product positive and maximal.
                let worst: Vec<u8> = q16
                    .iter()
                    .map(|q| (if *q < 0 { -127i8 } else { 127 }) as u8)
                    .collect();
                for (kernel, name) in available() {
                    assert_eq!(
                        i64::from(kernel(&q16, &worst)),
                        exact(&q16, &worst),
                        "{name}"
                    );
                }
            }
        }
    }

    /// The score is the cosine the f32 reference computes, to the precision int8 vectors
    /// and a 16-bit query carry.
    #[test]
    fn an_int8_score_tracks_the_f32_cosine() {
        let dim = 256;
        let mut random = Random(23);
        let vectors: Vec<Vec<f32>> = (0..200).map(|_| random.unit(dim)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let codec = Codec::calibrate_i8_sym_dim(&refs, 1.0).unwrap();
        let reference = Codec::f32(dim).unwrap();
        let query = random.unit(dim);
        let PreparedQuery::Int8 { q16, inv_scale } = PreparedQuery::new(&query, &codec).unwrap()
        else {
            panic!()
        };
        let PreparedQuery::Float { q } = PreparedQuery::new(&query, &reference).unwrap() else {
            panic!()
        };
        let (mut bytes, mut floats) = (vec![0u8; dim], vec![0u8; dim * 4]);
        for vector in &vectors {
            codec.encode(vector, &mut bytes);
            reference.encode(vector, &mut floats);
            let exact: f32 = query.iter().zip(vector).map(|(a, b)| a * b).sum();
            let float = dot_f32(&q, &floats);
            assert!((float - exact).abs() < 1e-5, "{float} vs {exact}");
            let int8 = dot_scalar(&q16, &bytes) as f32 * inv_scale;
            assert!((int8 - exact).abs() < 0.02, "{int8} vs {exact}");
        }
    }

    /// `i8-sym-vec`: the query is quantized as it is, under the same bound, and a score is
    /// the vector's scale times the exact integer sum — the same `f32` from every kernel,
    /// and the cosine of the decoded vector to the precision a 16-bit query carries; the
    /// cosine of the vector itself to the precision of its 8-bit codes.
    #[test]
    fn a_per_vector_score_is_the_same_on_every_kernel_and_tracks_the_cosine() {
        let dim = 256;
        let codec = Codec::i8_sym_vec(dim).unwrap();
        let mut random = Random(24);
        let mut queries: Vec<Vec<f32>> = (0..10).map(|_| random.unit(dim)).collect();
        queries.push({
            let mut q = vec![1e-4f32; dim];
            q[3] = 1.0;
            q
        });
        let vectors: Vec<Vec<f32>> = (0..200).map(|_| random.unit(dim)).collect();
        let mut codes = vec![0u8; dim];
        for query in &queries {
            let PreparedQuery::Int8PerVector { q16, inv_scale } =
                PreparedQuery::new(query, &codec).unwrap()
            else {
                panic!("a per-vector codec prepares a per-vector query");
            };
            let bound: i64 = q16.iter().map(|q| i64::from(q.unsigned_abs()) * 127).sum();
            assert!(bound <= i64::from(i32::MAX), "bound {bound}");
            let worst: Vec<u8> = q16
                .iter()
                .map(|q| (if *q < 0 { -127i8 } else { 127 }) as u8)
                .collect();
            for vector in vectors
                .iter()
                .map(Vec::as_slice)
                .chain([worst_vector(&worst)].iter().map(Vec::as_slice))
            {
                let scale = codec.encode(vector, &mut codes).scale.unwrap();
                let scores: Vec<u32> = available()
                    .into_iter()
                    .map(|(kernel, _)| {
                        ((kernel(&q16, &codes) as f32 * scale) * inv_scale).to_bits()
                    })
                    .collect();
                assert!(
                    scores.windows(2).all(|pair| pair[0] == pair[1]),
                    "{scores:?}"
                );
                let score = f32::from_bits(scores[0]);
                let mut decoded = vec![0f32; dim];
                codec.decode(&codes, Some(scale), &mut decoded);
                let of_decoded: f64 = query
                    .iter()
                    .zip(&decoded)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                let exact: f64 = query
                    .iter()
                    .zip(vector)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                assert!(
                    (f64::from(score) - of_decoded).abs() < 2e-3,
                    "{score} vs {of_decoded}"
                );
                assert!(
                    (f64::from(score) - exact).abs() < 0.02,
                    "{score} vs {exact}"
                );
            }
        }
    }

    /// A unit vector whose codes are `codes`: what the worst case looks like as floats.
    fn worst_vector(codes: &[u8]) -> Vec<f32> {
        let raw: Vec<f32> = codes.iter().map(|byte| f32::from(*byte as i8)).collect();
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    /// Every 4-bit kernel gives the scalar one's integer at every width from 1 to 80 bytes,
    /// on random weights and codes and on the extremes: ±32767 against 15.
    #[test]
    fn every_nibble_kernel_computes_the_scalar_kernels_integer() {
        let kernels = available_nibbles();
        assert!(kernels.iter().any(|(_, name)| *name == select_nibbles().1));
        let mut random = Random(25);
        for bytes in 1..=80 {
            for extreme in [false, true] {
                let mut weight = || -> i16 {
                    if extreme {
                        if random.next().is_multiple_of(2) {
                            32767
                        } else {
                            -32767
                        }
                    } else {
                        (random.next() % 65535) as i32 as i16
                    }
                };
                let lo: Vec<i16> = (0..bytes).map(|_| weight()).collect();
                let hi: Vec<i16> = (0..bytes).map(|_| weight()).collect();
                let packed: Vec<u8> = (0..bytes)
                    .map(|_| {
                        if extreme {
                            0xFF
                        } else {
                            (random.next() % 256) as u8
                        }
                    })
                    .collect();
                let reference = nibbles_scalar(&lo, &hi, &packed);
                let exact: i64 = lo
                    .iter()
                    .zip(&hi)
                    .zip(&packed)
                    .map(|((lo, hi), byte)| {
                        i64::from(*lo) * i64::from(byte & 15)
                            + i64::from(*hi) * i64::from(byte >> 4)
                    })
                    .sum();
                assert_eq!(i64::from(reference), exact, "{bytes} bytes");
                for (kernel, name) in &kernels {
                    assert_eq!(kernel(&lo, &hi, &packed), reference, "{name} at {bytes}");
                }
            }
        }
    }

    /// A rotated query: its head and tail each fit their bound, the worst vectors sum
    /// exactly on every kernel, and its score is the cosine of the query with the decoded
    /// vector, the same `f32` from every kernel.
    #[test]
    fn a_rotated_score_is_the_same_on_every_kernel_and_is_the_decoded_cosine() {
        let dim = 64;
        let mut random = Random(26);
        let corpus: Vec<Vec<f32>> = (0..400).map(|_| random.unit(dim)).collect();
        let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();
        let codec = Codec::fit_rot_i8i4(&refs, 32, 16).unwrap();
        let mut queries: Vec<Vec<f32>> = (0..8).map(|_| random.unit(dim)).collect();
        queries.push({
            let mut q = vec![1e-4f32; dim];
            q[5] = 1.0;
            q
        });
        let mut codes = vec![0u8; codec.bytes_per_vector()];
        let mut decoded = vec![0f32; dim];
        for query in &queries {
            let PreparedQuery::Rotated {
                dim: width,
                head,
                inv_head,
                tail_lo,
                tail_hi,
                inv_tail,
                constant,
            } = PreparedQuery::new(query, &codec).unwrap()
            else {
                panic!("a rotated codec prepares a rotated query");
            };
            assert_eq!(width, dim);
            let bound8: i64 = head.iter().map(|q| i64::from(q.unsigned_abs()) * 127).sum();
            let bound4: i64 = tail_lo
                .iter()
                .chain(tail_hi.iter())
                .map(|q| i64::from(q.unsigned_abs()) * 15)
                .sum();
            assert!(bound8 <= i64::from(i32::MAX) && bound4 <= i64::from(i32::MAX));
            let worst_head: Vec<u8> = head
                .iter()
                .map(|q| (if *q < 0 { -127i8 } else { 127 }) as u8)
                .collect();
            for (kernel, name) in available() {
                let exact: i64 = head
                    .iter()
                    .zip(&worst_head)
                    .map(|(q, v)| i64::from(*q) * i64::from(*v as i8))
                    .sum();
                assert_eq!(i64::from(kernel(&head, &worst_head)), exact, "{name}");
            }
            for vector in &corpus[..50] {
                let scale = codec.encode(vector, &mut codes).scale.unwrap();
                let (head_codes, packed) = codes.split_at(head.len());
                let scores: Vec<u32> = available()
                    .into_iter()
                    .zip(available_nibbles())
                    .map(|((dot, _), (nibbles, _))| {
                        let acc8 = dot(&head, head_codes);
                        let acc4 = nibbles(&tail_lo, &tail_hi, packed);
                        let score =
                            acc8 as f32 * scale * inv_head + (constant + acc4 as f32 * inv_tail);
                        score.to_bits()
                    })
                    .collect();
                assert!(
                    scores.windows(2).all(|pair| pair[0] == pair[1]),
                    "{scores:?}"
                );
                let score = f64::from(f32::from_bits(scores[0]));
                codec.decode(&codes, Some(scale), &mut decoded);
                let of_decoded: f64 = query
                    .iter()
                    .zip(&decoded)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                assert!((score - of_decoded).abs() < 2e-3, "{score} vs {of_decoded}");
            }
        }
    }

    #[test]
    fn ranks_order_as_the_floats_do_and_invert() {
        let values = [
            f32::NEG_INFINITY,
            -1.0,
            -1e-30,
            -0.0,
            0.0,
            1e-30,
            0.5,
            1.0,
            f32::INFINITY,
        ];
        for pair in values.windows(2) {
            assert!(ordered(pair[0]) < ordered(pair[1]), "{pair:?}");
        }
        for value in values {
            assert_eq!(unordered(ordered(value)).to_bits(), value.to_bits());
        }
    }

    #[test]
    fn a_query_of_the_wrong_width_is_refused() {
        let codec = Codec::f32(4).unwrap();
        assert!(matches!(
            PreparedQuery::new(&[1.0; 3], &codec),
            Err(VectorStoreError::DimensionMismatch { .. })
        ));
    }
}
