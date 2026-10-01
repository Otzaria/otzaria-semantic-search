//! How a unit vector becomes the bytes of one slot, and back.
//!
//! The registry is a closed list, as the recipe versions are: a segment names its codec by
//! number, and a number with no implementation here is refused rather than guessed at.
//!
//! | id | name (= `store.vector_precision`) | bytes per slot | parameters |
//! |---|---|---:|---|
//! | `0x0001` | `f32` | 4 × dim | none — the reference, and tests |
//! | `0x0101` | `i8-sym-dim` | dim | one scale per dimension, fixed for a codec epoch |
//! | `0x0102` | `i8-sym-vec` | dim + 4 | reserved: a scale per vector |
//! | `0x02xx` | binary or PQ | — | reserved for an ancillary first-pass tier |
//!
//! **A codec epoch** is one set of parameters. Every segment of a set shares it — the set
//! records its `codec_params_sha256` and refuses a delta from another — so the int8 bytes
//! of every segment mean the same thing and compaction is a byte copy. A study of other
//! codecs can still choose freely later: the build machine keeps an f32 copy of every
//! vector, re-emits any codec from it, and a new epoch is a new base.
//!
//! `i8-sym-dim`: `v_d = clamp(round_half_away(x_d / S_d · 127), −127, 127)`, in `f32`.
//! A component past its dimension's scale is *clipped*, and [`Codec::encode`] counts the
//! clipped components so a build can hold the rate to its gate.

use crate::semantic::oxv::format::le_u32;
use sha2::{Digest, Sha256};

/// The reference codec: the vector as it is.
pub const CODEC_F32: u16 = 0x0001;
/// Symmetric int8 with one scale per dimension: the default.
pub const CODEC_I8_SYM_DIM: u16 = 0x0101;
/// Symmetric int8 with one scale per vector. Reserved; not implemented.
pub const CODEC_I8_SYM_VEC: u16 = 0x0102;

/// The layout version of every codec's parameters section.
const PARAMS_VERSION: u32 = 1;

/// A codec and its parameters — one codec epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct Codec {
    dim: usize,
    kind: CodecKind,
    params: Box<[u8]>,
    params_sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq)]
enum CodecKind {
    F32,
    I8SymDim {
        /// What clipped one component in `1 − clip_q` of the calibration set. Recorded, not
        /// used: the scales are what quantize.
        clip_q: f32,
        scales: Box<[f32]>,
    },
}

impl Codec {
    /// The `f32` codec at width `dim`.
    pub fn f32(dim: usize) -> Result<Self, String> {
        let dim = valid_dim(dim)?;
        let mut params = Vec::with_capacity(8);
        params.extend_from_slice(&PARAMS_VERSION.to_le_bytes());
        params.extend_from_slice(&(dim as u32).to_le_bytes());
        Ok(Self::with_params(dim, CodecKind::F32, params))
    }

    /// The `i8-sym-dim` codec with these per-dimension scales.
    ///
    /// Every scale has to be finite and positive: a zero would divide, and a negative one
    /// would flip the sign of a dimension in every stored vector.
    pub fn i8_sym_dim(scales: Vec<f32>, clip_q: f32) -> Result<Self, String> {
        let dim = valid_dim(scales.len())?;
        if let Some((index, scale)) = scales
            .iter()
            .enumerate()
            .find(|(_, scale)| !(scale.is_finite() && **scale > 0.0))
        {
            return Err(format!(
                "the scale of dimension {index} is {scale}, and a scale must be finite and \
                 positive"
            ));
        }
        if !(clip_q.is_finite() && clip_q > 0.0 && clip_q <= 1.0) {
            return Err(format!("clip_q is {clip_q}, and it must be in (0, 1]"));
        }
        let mut params = Vec::with_capacity(12 + 4 * dim);
        params.extend_from_slice(&PARAMS_VERSION.to_le_bytes());
        params.extend_from_slice(&(dim as u32).to_le_bytes());
        params.extend_from_slice(&clip_q.to_le_bytes());
        for scale in &scales {
            params.extend_from_slice(&scale.to_le_bytes());
        }
        Ok(Self::with_params(
            dim,
            CodecKind::I8SymDim {
                clip_q,
                scales: scales.into_boxed_slice(),
            },
            params,
        ))
    }

    /// Calibrate `i8-sym-dim` on a set of vectors: each dimension's scale is the exact
    /// `clip_q` quantile of its absolute values — the ⌊clip_q · (n − 1)⌋-th smallest of the
    /// `n`, the "lower" rule, which no rounding of `clip_q` can move by a rank — so about
    /// `1 − clip_q` of that dimension's components is clipped. `clip_q = 1` takes the
    /// maximum, and clips nothing in the calibration set.
    ///
    /// A dimension that is zero in every vector gets scale 1: any scale encodes zero
    /// exactly, and a zero scale is not one.
    pub fn calibrate_i8_sym_dim(vectors: &[&[f32]], clip_q: f32) -> Result<Self, String> {
        let dim = vectors
            .first()
            .map(|vector| vector.len())
            .ok_or("there are no vectors to calibrate on")?;
        if let Some(vector) = vectors.iter().find(|vector| vector.len() != dim) {
            return Err(format!(
                "the calibration vectors are {dim} and {} wide",
                vector.len()
            ));
        }
        if !(clip_q.is_finite() && clip_q > 0.0 && clip_q <= 1.0) {
            return Err(format!("clip_q is {clip_q}, and it must be in (0, 1]"));
        }
        let n = vectors.len();
        let rank = ((f64::from(clip_q) * (n - 1) as f64).floor() as usize).min(n - 1);
        let mut column = vec![0f32; n];
        let mut scales = Vec::with_capacity(dim);
        for d in 0..dim {
            for (slot, vector) in column.iter_mut().zip(vectors) {
                *slot = vector[d].abs();
            }
            let (_, quantile, _) = column.select_nth_unstable_by(rank, f32::total_cmp);
            let scale = *quantile;
            scales.push(if scale > 0.0 && scale.is_finite() {
                scale
            } else {
                1.0
            });
        }
        Self::i8_sym_dim(scales, clip_q)
    }

    /// The codec a segment declares: its id, its width and its parameters section.
    pub fn from_params(id: u16, dim: usize, params: &[u8]) -> Result<Self, String> {
        if params.len() < 8 {
            return Err(format!(
                "its codec parameters are {} byte(s), too short for a version and a width",
                params.len()
            ));
        }
        let version = le_u32(params, 0);
        if version != PARAMS_VERSION {
            return Err(format!(
                "its codec parameters are version {version}, and this build reads \
                 {PARAMS_VERSION}"
            ));
        }
        let declared = le_u32(params, 4) as usize;
        if declared != dim {
            return Err(format!(
                "its codec parameters are for width {declared}, and the header declares {dim}"
            ));
        }
        let codec = match id {
            CODEC_F32 => {
                if params.len() != 8 {
                    return Err("the f32 codec takes no parameters".to_string());
                }
                Self::f32(dim)?
            }
            CODEC_I8_SYM_DIM => {
                if params.len() != 12 + 4 * dim {
                    return Err(format!(
                        "i8-sym-dim parameters for width {dim} are {} bytes, not {}",
                        params.len(),
                        12 + 4 * dim
                    ));
                }
                let clip_q = f32::from_le_bytes(params[8..12].try_into().expect("four bytes"));
                let scales = params[12..]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect();
                Self::i8_sym_dim(scales, clip_q)?
            }
            CODEC_I8_SYM_VEC => {
                return Err("codec i8-sym-vec (0x0102) is reserved and not implemented".into())
            }
            other => {
                return Err(format!(
                    "codec {other:#06x} is not one this build implements"
                ))
            }
        };
        debug_assert_eq!(&*codec.params, params);
        Ok(codec)
    }

    fn with_params(dim: usize, kind: CodecKind, params: Vec<u8>) -> Self {
        let params_sha256 = Sha256::digest(&params).into();
        Self {
            dim,
            kind,
            params: params.into_boxed_slice(),
            params_sha256,
        }
    }

    pub fn id(&self) -> u16 {
        match self.kind {
            CodecKind::F32 => CODEC_F32,
            CodecKind::I8SymDim { .. } => CODEC_I8_SYM_DIM,
        }
    }

    /// The codec's name, which is what a store identity declares as `vector_precision`.
    pub fn name(&self) -> &'static str {
        codec_name(self.id()).expect("an implemented codec has a name")
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The bytes one slot's vector takes.
    pub fn bytes_per_vector(&self) -> usize {
        match self.kind {
            CodecKind::F32 => 4 * self.dim,
            CodecKind::I8SymDim { .. } => self.dim,
        }
    }

    /// The CODEC_PARAMS section, byte for byte.
    pub fn params(&self) -> &[u8] {
        &self.params
    }

    /// SHA-256 of [`Self::params`]: the codec epoch, in 32 bytes.
    pub fn params_sha256(&self) -> [u8; 32] {
        self.params_sha256
    }

    /// The per-dimension scales of `i8-sym-dim`; `None` for `f32`.
    pub fn scales(&self) -> Option<&[f32]> {
        match &self.kind {
            CodecKind::F32 => None,
            CodecKind::I8SymDim { scales, .. } => Some(scales),
        }
    }

    /// The calibration quantile `i8-sym-dim` was built with; `None` for `f32`.
    pub fn clip_q(&self) -> Option<f32> {
        match &self.kind {
            CodecKind::F32 => None,
            CodecKind::I8SymDim { clip_q, .. } => Some(*clip_q),
        }
    }

    /// Encode `vector` into `out`, which must be [`Self::bytes_per_vector`] long, and
    /// return how many components were clipped.
    ///
    /// # Panics
    ///
    /// When either length is not the codec's.
    pub fn encode(&self, vector: &[f32], out: &mut [u8]) -> usize {
        assert_eq!(vector.len(), self.dim, "a vector of the codec's width");
        assert_eq!(
            out.len(),
            self.bytes_per_vector(),
            "a slot of the codec's width"
        );
        match &self.kind {
            CodecKind::F32 => {
                for (bytes, value) in out.as_chunks_mut::<4>().0.iter_mut().zip(vector) {
                    *bytes = value.to_le_bytes();
                }
                0
            }
            CodecKind::I8SymDim { scales, .. } => {
                let mut clipped = 0;
                for ((byte, value), scale) in out.iter_mut().zip(vector).zip(scales.iter()) {
                    let (quantized, clip) = quantize(*value, *scale);
                    clipped += usize::from(clip);
                    *byte = quantized as u8;
                }
                clipped
            }
        }
    }

    /// The vector `bytes` decodes to: exact for `f32`, `v_d · S_d / 127` for int8.
    pub fn decode(&self, bytes: &[u8], out: &mut [f32]) {
        assert_eq!(bytes.len(), self.bytes_per_vector());
        assert_eq!(out.len(), self.dim);
        match &self.kind {
            CodecKind::F32 => {
                for (value, bytes) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *value = f32::from_le_bytes(*bytes);
                }
            }
            CodecKind::I8SymDim { scales, .. } => {
                for ((value, byte), scale) in out.iter_mut().zip(bytes).zip(scales.iter()) {
                    *value = f32::from(*byte as i8) * scale / 127.0;
                }
            }
        }
    }
}

/// One component under one scale: `clamp(round_half_away(x / S · 127), −127, 127)`, and
/// whether the clamp cut it. `f32::round` rounds half away from zero, so this is the same
/// number on every machine.
fn quantize(value: f32, scale: f32) -> (i8, bool) {
    let scaled = (value / scale * 127.0).round();
    if scaled > 127.0 {
        (127, true)
    } else if scaled < -127.0 {
        (-127, true)
    } else if scaled.is_nan() {
        // Unreachable for a vector the runtime accepted, which is finite; a zero rather
        // than a guess if it ever is not.
        (0, true)
    } else {
        (scaled as i8, false)
    }
}

/// The name a codec id stands for in a store identity, when this build implements it.
pub fn codec_name(id: u16) -> Option<&'static str> {
    match id {
        CODEC_F32 => Some("f32"),
        CODEC_I8_SYM_DIM => Some("i8-sym-dim"),
        _ => None,
    }
}

fn valid_dim(dim: usize) -> Result<usize, String> {
    if dim == 0 || dim > u16::MAX as usize {
        return Err(format!(
            "a codec's width is between 1 and 65535, and this one is {dim}"
        ));
    }
    Ok(dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_rounds_half_away_from_zero_and_counts_what_it_clips() {
        // Scales of 127 make x / S · 127 exactly x, so 63.5 is an exact tie.
        let codec = Codec::i8_sym_dim(vec![127.0, 127.0, 127.0, 127.0, 0.5, 2.0], 1.0).unwrap();
        let vector = [63.5, -63.5, 127.0, -127.0, 0.75, 0.0];
        let mut out = [0u8; 6];
        assert_eq!(codec.encode(&vector, &mut out), 1);
        let values: Vec<i8> = out.iter().map(|byte| *byte as i8).collect();
        // ±63.5 rounds away from zero; ±127 is the scale itself and is not clipped; 0.75 is
        // past its scale of 0.5 and clips to 127.
        assert_eq!(values, vec![64, -64, 127, -127, 127, 0]);

        let mut back = [0f32; 6];
        codec.decode(&out, &mut back);
        assert_eq!(back[3], -127.0);
        assert_eq!(back[4], 0.5);
    }

    #[test]
    fn parameters_round_trip_and_name_the_epoch() {
        let codec = Codec::i8_sym_dim(vec![0.25, 0.5, 0.125], 0.9999).unwrap();
        let read = Codec::from_params(CODEC_I8_SYM_DIM, 3, codec.params()).unwrap();
        assert_eq!(read, codec);
        assert_eq!(read.name(), "i8-sym-dim");
        assert_eq!(read.bytes_per_vector(), 3);

        let other = Codec::i8_sym_dim(vec![0.25, 0.5, 0.126], 0.9999).unwrap();
        assert_ne!(other.params_sha256(), codec.params_sha256());

        let reference = Codec::f32(3).unwrap();
        assert_eq!(
            Codec::from_params(CODEC_F32, 3, reference.params()).unwrap(),
            reference
        );
        assert_eq!(reference.bytes_per_vector(), 12);
        assert_eq!(reference.name(), "f32");
    }

    #[test]
    fn a_codec_or_a_parameter_this_build_cannot_use_is_refused() {
        let codec = Codec::i8_sym_dim(vec![0.25, 0.5], 1.0).unwrap();
        assert!(Codec::from_params(CODEC_I8_SYM_DIM, 3, codec.params()).is_err());
        assert!(Codec::from_params(CODEC_I8_SYM_VEC, 2, codec.params())
            .unwrap_err()
            .contains("reserved"));
        assert!(Codec::from_params(0x0201, 2, codec.params()).is_err());
        assert!(Codec::from_params(CODEC_I8_SYM_DIM, 2, &codec.params()[..10]).is_err());
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(Codec::i8_sym_dim(vec![1.0, scale], 1.0).is_err(), "{scale}");
        }
        assert!(Codec::i8_sym_dim(vec![1.0], 0.0).is_err());
        assert!(Codec::f32(0).is_err());
    }

    #[test]
    fn calibration_takes_the_exact_quantile_of_each_dimension() {
        let vectors: Vec<Vec<f32>> = (1..=100)
            .map(|i| vec![i as f32 / 100.0, -(i as f32) / 50.0, 0.0])
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let max = Codec::calibrate_i8_sym_dim(&refs, 1.0).unwrap();
        assert_eq!(max.scales().unwrap(), &[1.0, 2.0, 1.0]);

        let q99 = Codec::calibrate_i8_sym_dim(&refs, 0.99).unwrap();
        assert_eq!(q99.scales().unwrap(), &[0.99, 1.98, 1.0]);
        assert_eq!(q99.clip_q(), Some(0.99));
        let mut out = [0u8; 3];
        // The one vector past the 0.99 quantile in both dimensions clips in both.
        assert_eq!(q99.encode(&vectors[99], &mut out), 2);
        assert_eq!(q99.encode(&vectors[98], &mut out), 0);
    }
}
