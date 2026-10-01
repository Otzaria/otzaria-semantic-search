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

/// A query, prepared once for the codec epoch of the set it scans.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PreparedQuery {
    Int8 { q16: Box<[i16]>, inv_scale: f32 },
    Float { q: Box<[f32]> },
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
        Ok(match codec.scales() {
            None => Self::Float { q: query.into() },
            Some(scales) => {
                let (q16, inv_scale) = quantize_query(query, scales);
                Self::Int8 { q16, inv_scale }
            }
        })
    }
}

/// `q16` and `1 / t`, per the module documentation. In `f64`, then checked in integers: the
/// bound is the guarantee, so it is verified rather than trusted to the rounding of `t`.
fn quantize_query(query: &[f32], scales: &[f32]) -> (Box<[i16]>, f32) {
    let w: Vec<f64> = query
        .iter()
        .zip(scales)
        .map(|(q, s)| f64::from(*q) * f64::from(*s) / 127.0)
        .collect();
    let max = w.iter().fold(0.0f64, |max, x| max.max(x.abs()));
    let sum: f64 = w.iter().map(|x| x.abs()).sum();
    if !(max > 0.0 && max.is_finite() && sum.is_finite()) {
        // A zero query scores everything zero; the runtime never hands one over.
        return (vec![0; w.len()].into_boxed_slice(), 0.0);
    }
    let headroom = f64::from(i32::MAX) - 127.0 * w.len() as f64 / 2.0;
    let mut t = (32767.0 / max).min(headroom / (127.0 * sum));
    loop {
        let q16: Vec<i16> = w
            .iter()
            .map(|x| (x * t).round().clamp(-32767.0, 32767.0) as i16)
            .collect();
        let bound: i64 = q16.iter().map(|q| i64::from(q.unsigned_abs()) * 127).sum();
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
