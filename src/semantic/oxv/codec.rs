//! How a unit vector becomes the bytes of one slot, and back.
//!
//! The registry is a closed list, as the recipe versions are: a segment names its codec by
//! number, and a number with no implementation here is refused rather than guessed at.
//!
//! | id | name (= `store.vector_precision`) | bytes per slot | parameters |
//! |---|---|---:|---|
//! | `0x0001` | `f32` | 4 × dim | none — the reference, and tests |
//! | `0x0101` | `i8-sym-dim` | dim | one scale per dimension, fixed for a codec epoch |
//! | `0x0102` | `i8-sym-vec` | dim, + 4 in VECTOR_SCALES | none: a scale per vector — the default |
//! | `0x0103` | `rot192-i8i4` | head + tail / 2, + 4 in VECTOR_SCALES | a rotation and the tail's ranges |
//! | `0x02xx` | binary or PQ | — | reserved for an ancillary first-pass tier |
//!
//! **A codec epoch** is one set of parameters. Every segment of a set shares it — the set
//! records its `codec_params_sha256` and refuses a delta from another — so the int8 bytes
//! of every segment mean the same thing and compaction is a byte copy. A study of other
//! codecs can still choose freely later: the build machine keeps an f32 copy of every
//! vector, re-emits any codec from it, and a new epoch is a new base.
//!
//! `i8-sym-vec`, the default: each vector has its own scale `s = max_d |x_d| / 127`, and
//! `v_d = clamp(round_half_away(x_d / s), −127, 127)` in `f32`; a vector decodes as `v · s`.
//! Its codes fill VECTORS — `dim` bytes a slot, so rows stay aligned — and its scales a
//! separate VECTOR_SCALES array of `f32`s. Nothing is clipped, and a codec epoch is only a
//! width: no calibration, and a delta needs nothing of the base's. On the library's
//! vectors it keeps recall@10 above 0.99 against full precision for int8 and fp32 queries
//! (0.9918 / 0.9914), where `i8-sym-dim` reaches 0.9866.
//!
//! `rot192-i8i4`, for format-3 sets: `y = R · x`, `R` the `out_dim × dim` rotation onto
//! the leading eigenvectors of the corpus's second moment (see [`Codec::fit_rot_i8i4`]).
//! The first `head` components (128) are divided by a frozen per-component scale `σ_h`
//! and stored as `i8-sym-vec` — `s = max |y_h / σ_h| / 127` in VECTOR_SCALES — and the
//! next `tail` (64) as 4-bit codes, two to a byte (the even component in the low nibble),
//! affine over a range frozen per component:
//! `q_d = clamp(round_half_away((y_d − lo_d) / (hi_d − lo_d) · 15), 0, 15)`. The other
//! components are dropped. 160 bytes and a scale a vector; it decodes as `Rᵀ · ŷ`, so a
//! score stays a cosine in the model's space. A tail component past its range clips.
//!
//! Measured on 400k vectors of the v31 base against the exact scan of their `i8-sym-vec`
//! decodings, 4000 queries: `σ_h = √std_h` lifts recall@10 from 0.979 to 0.982 (no scale,
//! or `σ_h = std_h`, 0.980); the affine tail is the float tail's recall, where a symmetric
//! 4-bit tail — per vector or per component — loses a further 0.002–0.004.
//!
//! `i8-sym-dim`: `v_d = clamp(round_half_away(x_d / S_d · 127), −127, 127)`, in `f32`.
//! A component past its dimension's scale is *clipped*, and [`Codec::encode`] counts the
//! clipped components so a build can hold the rate to its gate.

use crate::semantic::oxv::format::le_u32;
use sha2::{Digest, Sha256};

/// The reference codec: the vector as it is.
pub const CODEC_F32: u16 = 0x0001;
/// Symmetric int8 with one scale per dimension.
pub const CODEC_I8_SYM_DIM: u16 = 0x0101;
/// Symmetric int8 with one scale per vector: the default.
pub const CODEC_I8_SYM_VEC: u16 = 0x0102;
/// A rotation, then int8 for the head and 4 bits for the tail.
pub const CODEC_ROT_I8I4: u16 = 0x0103;
/// Its name.
pub const ROT_I8I4_NAME: &str = "rot192-i8i4";
/// Its layout: components kept in int8, and in 4 bits after them.
pub const ROT_HEAD: usize = 128;
pub const ROT_TAIL: usize = 64;

/// What a build asks for: the codec, before any calibration it needs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CodecSpec {
    /// `i8-sym-vec`, the default.
    #[default]
    I8SymVec,
    /// `i8-sym-dim`, its scales calibrated at the `clip_q` quantile of each dimension.
    I8SymDim { clip_q: f32 },
    /// `f32`.
    F32,
}

impl CodecSpec {
    /// The name a store identity declares for the codec.
    pub fn name(self) -> &'static str {
        match self {
            Self::I8SymVec => "i8-sym-vec",
            Self::I8SymDim { .. } => "i8-sym-dim",
            Self::F32 => "f32",
        }
    }

    /// Parse a name — and, for `i8-sym-dim`, its quantile.
    pub fn parse(name: &str, clip_q: f32) -> Result<Self, String> {
        match name {
            "i8-sym-vec" => Ok(Self::I8SymVec),
            "i8-sym-dim" => Ok(Self::I8SymDim { clip_q }),
            "f32" => Ok(Self::F32),
            ROT_I8I4_NAME => Err(format!(
                "{ROT_I8I4_NAME} is fitted, not named: fit it with fit-rotation and give the \
                 file to assemble with --rotation"
            )),
            other => Err(format!(
                "{other:?} is not a codec; the codecs are i8-sym-vec, i8-sym-dim and f32"
            )),
        }
    }

    /// The codec at width `dim`, calibrated on `vectors` where it needs to be.
    pub fn build(self, dim: usize, vectors: &[&[f32]]) -> Result<Codec, String> {
        match self {
            Self::I8SymVec => Codec::i8_sym_vec(dim),
            Self::I8SymDim { clip_q } => Codec::calibrate_i8_sym_dim(vectors, clip_q),
            Self::F32 => Codec::f32(dim),
        }
    }
}

/// What [`Codec::encode`] produced beside the bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Encoded {
    /// Components the codec's scale cut.
    pub clipped: usize,
    /// The vector's own scale, for a codec with one.
    pub scale: Option<f32>,
}

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
    I8SymVec,
    RotI8I4(Box<Rotated>),
    I8SymDim {
        /// What clipped one component in `1 − clip_q` of the calibration set. Recorded, not
        /// used: the scales are what quantize.
        clip_q: f32,
        scales: Box<[f32]>,
    },
}

/// The parameters of `rot192-i8i4`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Rotated {
    pub head: usize,
    pub tail: usize,
    /// Each head component's spread: it is quantized as `y / scale`, so every head
    /// component spans the codes alike.
    pub scale: Box<[f32]>,
    /// Each tail component's range: `lo` encodes as 0, `hi` as 15.
    pub lo: Box<[f32]>,
    pub hi: Box<[f32]>,
    /// `head + tail` rows of `dim` components, row-major.
    pub rotation: Box<[f32]>,
}

impl Rotated {
    pub(crate) fn out_dim(&self) -> usize {
        self.head + self.tail
    }

    /// What one code of tail component `d` is worth.
    pub(crate) fn step(&self, d: usize) -> f32 {
        (self.hi[d] - self.lo[d]) / 15.0
    }
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

    /// The `i8-sym-vec` codec at width `dim`.
    pub fn i8_sym_vec(dim: usize) -> Result<Self, String> {
        let dim = valid_dim(dim)?;
        let mut params = Vec::with_capacity(8);
        params.extend_from_slice(&PARAMS_VERSION.to_le_bytes());
        params.extend_from_slice(&(dim as u32).to_le_bytes());
        Ok(Self::with_params(dim, CodecKind::I8SymVec, params))
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

    /// `rot192-i8i4` with this rotation — `head + tail` rows of `dim` — head scales and
    /// tail ranges.
    ///
    /// The params section is `{version, dim, out_dim, head, tail}` as `u32`s, then `scale`
    /// (`head` `f32`s), `lo` and `hi` (`tail` `f32`s each), then the rotation (`out_dim × dim`
    /// `f32`s, row-major): all of it is the epoch, so its SHA-256 freezes the matrix.
    pub fn rot_i8i4(
        dim: usize,
        head: usize,
        tail: usize,
        scale: Vec<f32>,
        lo: Vec<f32>,
        hi: Vec<f32>,
        rotation: Vec<f32>,
    ) -> Result<Self, String> {
        let dim = valid_dim(dim)?;
        if head == 0 || !tail.is_multiple_of(2) || head + tail > dim {
            return Err(format!(
                "a rotated layout of {head} int8 and {tail} 4-bit components does not fit width \
                 {dim}: the head is not empty, the tail is even, and both fit"
            ));
        }
        if scale.len() != head
            || lo.len() != tail
            || hi.len() != tail
            || rotation.len() != (head + tail) * dim
        {
            return Err(format!(
                "a rotated codec of {head} + {tail} components of width {dim} has {head} scales, \
                 {tail} ranges and {} rotation entries",
                (head + tail) * dim
            ));
        }
        if let Some(d) = (0..head).find(|&d| !(scale[d].is_finite() && scale[d] > 0.0)) {
            return Err(format!(
                "head component {d} has scale {}, and a scale is finite and positive",
                scale[d]
            ));
        }
        if let Some(d) =
            (0..tail).find(|&d| !(lo[d].is_finite() && hi[d].is_finite() && lo[d] < hi[d]))
        {
            return Err(format!(
                "tail component {d} ranges over [{}, {}], and a range is finite and not empty",
                lo[d], hi[d]
            ));
        }
        if rotation.iter().any(|value| !value.is_finite()) {
            return Err("the rotation holds a value that is not finite".to_string());
        }
        let mut params = Vec::with_capacity(20 + 4 * head + 8 * tail + 4 * rotation.len());
        for value in [
            PARAMS_VERSION,
            dim as u32,
            (head + tail) as u32,
            head as u32,
            tail as u32,
        ] {
            params.extend_from_slice(&value.to_le_bytes());
        }
        for value in scale.iter().chain(&lo).chain(&hi).chain(&rotation) {
            params.extend_from_slice(&value.to_le_bytes());
        }
        Ok(Self::with_params(
            dim,
            CodecKind::RotI8I4(Box::new(Rotated {
                head,
                tail,
                scale: scale.into_boxed_slice(),
                lo: lo.into_boxed_slice(),
                hi: hi.into_boxed_slice(),
                rotation: rotation.into_boxed_slice(),
            })),
            params,
        ))
    }

    /// Fit `rot192-i8i4` on `vectors`: the rotation onto the `head + tail` leading
    /// eigenvectors of their uncentered second moment `E[x xᵀ]` (the `rotation` module), each
    /// head component's scale as the square root of its standard deviation over them, and
    /// each tail component's range as the least and greatest it takes. The same vectors in the same order fit the same bytes.
    pub fn fit_rot_i8i4(vectors: &[&[f32]], head: usize, tail: usize) -> Result<Self, String> {
        use super::rotation::{principal_rows, rotate, second_moment};
        let dim = vectors
            .first()
            .map(|vector| vector.len())
            .ok_or("there are no vectors to fit on")?;
        if let Some(vector) = vectors.iter().find(|vector| vector.len() != dim) {
            return Err(format!(
                "the vectors to fit on are {dim} and {} wide",
                vector.len()
            ));
        }
        if head + tail > dim {
            return Err(format!("{head} + {tail} components do not fit width {dim}"));
        }
        let moment = second_moment(vectors.iter().copied(), dim);
        let rotation = principal_rows(moment, dim, head + tail);
        let (mut lo, mut hi) = (vec![f32::INFINITY; tail], vec![f32::NEG_INFINITY; tail]);
        let (mut sum, mut squares) = (vec![0f64; head], vec![0f64; head]);
        let mut y = vec![0f64; head + tail];
        for vector in vectors {
            rotate(&rotation, vector, &mut y);
            for (d, value) in y[..head].iter().enumerate() {
                let value = f64::from(*value as f32);
                sum[d] += value;
                squares[d] += value * value;
            }
            for (d, value) in y[head..].iter().enumerate() {
                let value = *value as f32;
                lo[d] = lo[d].min(value);
                hi[d] = hi[d].max(value);
            }
        }
        let n = vectors.len() as f64;
        let scale = sum
            .iter()
            .zip(&squares)
            .map(|(sum, squares)| {
                let mean = sum / n;
                let spread = (squares / n - mean * mean).max(0.0).sqrt().sqrt() as f32;
                // A component that never varies still needs a scale to divide by.
                if spread > 0.0 && spread.is_finite() {
                    spread
                } else {
                    1.0
                }
            })
            .collect();
        for (lo, hi) in lo.iter_mut().zip(hi.iter_mut()) {
            // A component that never varies still needs a range to divide by.
            if *hi <= *lo {
                *hi = *lo + 1e-6;
            }
        }
        Self::rot_i8i4(dim, head, tail, scale, lo, hi, rotation)
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
                if params.len() != 8 {
                    return Err("the i8-sym-vec codec takes no parameters".to_string());
                }
                Self::i8_sym_vec(dim)?
            }
            CODEC_ROT_I8I4 => {
                if params.len() < 20 {
                    return Err("rot192-i8i4 parameters are too short for their layout".into());
                }
                let (out_dim, head, tail) = (
                    le_u32(params, 8) as usize,
                    le_u32(params, 12) as usize,
                    le_u32(params, 16) as usize,
                );
                if head.checked_add(tail) != Some(out_dim) || out_dim > dim {
                    return Err(format!(
                        "rot192-i8i4 parameters declare {out_dim} components as {head} + {tail}"
                    ));
                }
                let expected = 20 + 4 * head + 8 * tail + 4 * out_dim * dim;
                if params.len() != expected {
                    return Err(format!(
                        "rot192-i8i4 parameters for {out_dim} × {dim} are {} bytes, not \
                         {expected}",
                        params.len()
                    ));
                }
                let floats: Vec<f32> = params[20..]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect();
                let (scale, rest) = floats.split_at(head);
                let (lo, rest) = rest.split_at(tail);
                let (hi, rotation) = rest.split_at(tail);
                Self::rot_i8i4(
                    dim,
                    head,
                    tail,
                    scale.to_vec(),
                    lo.to_vec(),
                    hi.to_vec(),
                    rotation.to_vec(),
                )?
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
            CodecKind::I8SymVec => CODEC_I8_SYM_VEC,
            CodecKind::I8SymDim { .. } => CODEC_I8_SYM_DIM,
            CodecKind::RotI8I4(_) => CODEC_ROT_I8I4,
        }
    }

    /// Whether each vector carries its own scale, stored in VECTOR_SCALES.
    pub fn has_vector_scales(&self) -> bool {
        matches!(self.kind, CodecKind::I8SymVec | CodecKind::RotI8I4(_))
    }

    /// Whether a component can fall outside what the codec encodes, and is counted when it
    /// does: `i8-sym-dim`'s scales and `rot192-i8i4`'s tail ranges.
    pub fn can_clip(&self) -> bool {
        matches!(
            self.kind,
            CodecKind::I8SymDim { .. } | CodecKind::RotI8I4(_)
        )
    }

    /// The components a clip rate is counted over: `dim`, or the rotated codec's tail.
    pub fn clippable_components(&self) -> usize {
        match &self.kind {
            CodecKind::RotI8I4(rotated) => rotated.tail,
            _ => self.dim,
        }
    }

    /// `rot192-i8i4`'s parameters; `None` for the other codecs.
    pub(crate) fn rotated(&self) -> Option<&Rotated> {
        match &self.kind {
            CodecKind::RotI8I4(rotated) => Some(rotated),
            _ => None,
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
        match &self.kind {
            CodecKind::F32 => 4 * self.dim,
            CodecKind::I8SymVec | CodecKind::I8SymDim { .. } => self.dim,
            CodecKind::RotI8I4(rotated) => rotated.head + rotated.tail / 2,
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

    /// The per-dimension scales of `i8-sym-dim`; `None` otherwise.
    pub fn scales(&self) -> Option<&[f32]> {
        match &self.kind {
            CodecKind::I8SymDim { scales, .. } => Some(scales),
            CodecKind::F32 | CodecKind::I8SymVec | CodecKind::RotI8I4(_) => None,
        }
    }

    /// The calibration quantile `i8-sym-dim` was built with; `None` otherwise.
    pub fn clip_q(&self) -> Option<f32> {
        match &self.kind {
            CodecKind::I8SymDim { clip_q, .. } => Some(*clip_q),
            CodecKind::F32 | CodecKind::I8SymVec | CodecKind::RotI8I4(_) => None,
        }
    }

    /// Encode `vector` into `out`, which must be [`Self::bytes_per_vector`] long, and
    /// return how many components were clipped and — for a codec with one — its scale.
    ///
    /// # Panics
    ///
    /// When either length is not the codec's.
    pub fn encode(&self, vector: &[f32], out: &mut [u8]) -> Encoded {
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
                Encoded {
                    clipped: 0,
                    scale: None,
                }
            }
            CodecKind::I8SymVec => Encoded {
                clipped: 0,
                scale: Some(encode_per_vector(vector, out)),
            },
            CodecKind::RotI8I4(rotated) => {
                let mut y64 = vec![0f64; rotated.out_dim()];
                super::rotation::rotate(&rotated.rotation, vector, &mut y64);
                let y: Vec<f32> = y64.iter().map(|value| *value as f32).collect();
                let (head, packed) = out.split_at_mut(rotated.head);
                let whitened: Vec<f32> = y[..rotated.head]
                    .iter()
                    .zip(rotated.scale.iter())
                    .map(|(y, scale)| y / scale)
                    .collect();
                let scale = encode_per_vector(&whitened, head);
                let mut clipped = 0;
                for (j, byte) in packed.iter_mut().enumerate() {
                    let mut nibbles = [0u8; 2];
                    for (half, nibble) in nibbles.iter_mut().enumerate() {
                        let d = 2 * j + half;
                        let (lo, hi) = (rotated.lo[d], rotated.hi[d]);
                        let code = ((y[rotated.head + d] - lo) / (hi - lo) * 15.0).round();
                        clipped += usize::from(!(0.0..=15.0).contains(&code));
                        *nibble = code.clamp(0.0, 15.0) as u8;
                    }
                    *byte = nibbles[0] | (nibbles[1] << 4);
                }
                Encoded {
                    clipped,
                    scale: Some(scale),
                }
            }
            CodecKind::I8SymDim { scales, .. } => {
                let mut clipped = 0;
                for ((byte, value), scale) in out.iter_mut().zip(vector).zip(scales.iter()) {
                    let (quantized, clip) = quantize(*value, *scale);
                    clipped += usize::from(clip);
                    *byte = quantized as u8;
                }
                Encoded {
                    clipped,
                    scale: None,
                }
            }
        }
    }

    /// The vector `bytes` decodes to: exact for `f32`, `v_d · S_d / 127` for `i8-sym-dim`,
    /// `v_d · s` for `i8-sym-vec` with the vector's own scale `s` (`scale`, required there
    /// and for `rot192-i8i4`, ignored elsewhere), and `Rᵀ · ŷ` for `rot192-i8i4`.
    pub fn decode(&self, bytes: &[u8], scale: Option<f32>, out: &mut [f32]) {
        assert_eq!(bytes.len(), self.bytes_per_vector());
        assert_eq!(out.len(), self.dim);
        match &self.kind {
            CodecKind::F32 => {
                for (value, bytes) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *value = f32::from_le_bytes(*bytes);
                }
            }
            CodecKind::I8SymVec => {
                let scale = scale.expect("an i8-sym-vec vector decodes with its scale");
                for (value, byte) in out.iter_mut().zip(bytes) {
                    *value = f32::from(*byte as i8) * scale;
                }
            }
            CodecKind::I8SymDim { scales, .. } => {
                for ((value, byte), scale) in out.iter_mut().zip(bytes).zip(scales.iter()) {
                    *value = f32::from(*byte as i8) * scale / 127.0;
                }
            }
            CodecKind::RotI8I4(rotated) => {
                let scale = scale.expect("a rot192-i8i4 vector decodes with its scale");
                let mut y = Vec::with_capacity(rotated.out_dim());
                y.extend(
                    bytes[..rotated.head]
                        .iter()
                        .zip(rotated.scale.iter())
                        .map(|(byte, spread)| f64::from(f32::from(*byte as i8) * scale * spread)),
                );
                for (j, byte) in bytes[rotated.head..].iter().enumerate() {
                    for (half, code) in [byte & 0x0F, byte >> 4].into_iter().enumerate() {
                        let d = 2 * j + half;
                        y.push(f64::from(rotated.lo[d] + f32::from(code) * rotated.step(d)));
                    }
                }
                super::rotation::unrotate(&rotated.rotation, &y, out);
            }
        }
    }
}

/// `i8-sym-vec` of `vector` into `out`: the codes, and the scale returned.
fn encode_per_vector(vector: &[f32], out: &mut [u8]) -> f32 {
    let max = vector
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    let scale = max / 127.0;
    for (byte, value) in out.iter_mut().zip(vector) {
        *byte = if scale > 0.0 {
            quantize_per_vector(*value, scale) as u8
        } else {
            0
        };
    }
    scale
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

/// One component under its vector's scale: `clamp(round_half_away(x / s), −127, 127)`.
/// The clamp only guards rounding: `|x| ≤ 127 · s` by the scale's definition.
fn quantize_per_vector(value: f32, scale: f32) -> i8 {
    (value / scale).round().clamp(-127.0, 127.0) as i8
}

/// The name a codec id stands for in a store identity, when this build implements it.
pub fn codec_name(id: u16) -> Option<&'static str> {
    match id {
        CODEC_F32 => Some("f32"),
        CODEC_I8_SYM_DIM => Some("i8-sym-dim"),
        CODEC_I8_SYM_VEC => Some("i8-sym-vec"),
        CODEC_ROT_I8I4 => Some(ROT_I8I4_NAME),
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
        assert_eq!(codec.encode(&vector, &mut out).clipped, 1);
        let values: Vec<i8> = out.iter().map(|byte| *byte as i8).collect();
        // ±63.5 rounds away from zero; ±127 is the scale itself and is not clipped; 0.75 is
        // past its scale of 0.5 and clips to 127.
        assert_eq!(values, vec![64, -64, 127, -127, 127, 0]);

        let mut back = [0f32; 6];
        codec.decode(&out, None, &mut back);
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
            .contains("no parameters"));
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
        assert_eq!(q99.encode(&vectors[99], &mut out).clipped, 2);
        assert_eq!(q99.encode(&vectors[98], &mut out).clipped, 0);
    }

    /// `i8-sym-vec`: the scale is the vector's largest component over 127, the largest
    /// component encodes as ±127 exactly, nothing clips, and a vector decodes to within half
    /// a step of itself in every component.
    #[test]
    fn a_scale_per_vector_encodes_every_vector_to_within_half_a_step() {
        let codec = Codec::i8_sym_vec(4).unwrap();
        assert_eq!(codec.name(), "i8-sym-vec");
        assert_eq!(codec.bytes_per_vector(), 4);
        assert!(codec.has_vector_scales() && codec.scales().is_none());
        assert_eq!(
            Codec::from_params(CODEC_I8_SYM_VEC, 4, codec.params()).unwrap(),
            codec
        );
        assert_eq!(
            codec.params_sha256(),
            Codec::i8_sym_vec(4).unwrap().params_sha256(),
            "an epoch is only a width: a delta needs nothing of its base's"
        );

        let vector = [0.5f32, -0.25, 0.125, -0.8];
        let mut out = [0u8; 4];
        let encoded = codec.encode(&vector, &mut out);
        let scale = encoded.scale.unwrap();
        assert_eq!(encoded.clipped, 0);
        assert_eq!(scale, 0.8 / 127.0);
        let codes: Vec<i8> = out.iter().map(|byte| *byte as i8).collect();
        assert_eq!(codes[3], -127, "the largest component is the scale itself");
        let mut back = [0f32; 4];
        codec.decode(&out, Some(scale), &mut back);
        for (value, original) in back.iter().zip(vector) {
            assert!(
                (value - original).abs() <= scale / 2.0 + 1e-7,
                "{value} {original}"
            );
        }

        let mut zero = [9u8; 4];
        let encoded = codec.encode(&[0.0; 4], &mut zero);
        assert_eq!((zero, encoded.scale), ([0; 4], Some(0.0)));
    }

    /// `rot192-i8i4`: its parameters round trip and name the epoch, matrix included; the same
    /// vectors fit the same bytes; a vector encodes to the same bytes every time, two tail
    /// codes a byte, decodes to its projection, and counts what its ranges clip.
    #[test]
    fn a_rotated_codec_fits_deterministically_and_encodes_its_tail() {
        use crate::semantic::oxv::rotation::{rotate, unrotate};
        use crate::semantic::oxv::testing::Random;
        let dim = 32;
        let mut random = Random(41);
        let vectors: Vec<Vec<f32>> = (0..500).map(|_| random.unit(dim)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let codec = Codec::fit_rot_i8i4(&refs, 16, 8).unwrap();
        let again = Codec::fit_rot_i8i4(&refs, 16, 8).unwrap();
        assert_eq!(codec.params(), again.params());
        assert_eq!(codec.params_sha256(), again.params_sha256());
        assert_eq!((codec.id(), codec.name()), (CODEC_ROT_I8I4, ROT_I8I4_NAME));
        assert_eq!(codec.bytes_per_vector(), 16 + 4);
        assert!(codec.has_vector_scales() && codec.can_clip());
        assert_eq!(codec.clippable_components(), 8);
        assert_eq!(codec.params().len(), 20 + 16 * 4 + 2 * 8 * 4 + 24 * dim * 4);
        assert_eq!(
            Codec::from_params(CODEC_ROT_I8I4, dim, codec.params()).unwrap(),
            codec
        );

        let mut other = codec.params().to_vec();
        let last = other.len() - 1;
        other[last] ^= 1;
        let other = Codec::from_params(CODEC_ROT_I8I4, dim, &other).unwrap();
        assert_ne!(other.params_sha256(), codec.params_sha256());
        assert!(Codec::from_params(CODEC_ROT_I8I4, dim + 1, codec.params()).is_err());
        assert!(Codec::from_params(CODEC_ROT_I8I4, dim, &codec.params()[..100]).is_err());
        let mut layout = codec.params().to_vec();
        layout[12..16].copy_from_slice(&17u32.to_le_bytes());
        assert!(Codec::from_params(CODEC_ROT_I8I4, dim, &layout).is_err());
        let rotation = vec![0.0; 24 * dim];
        let build = |scale: f32, tail: usize, lo: f32, rotation: Vec<f32>| {
            Codec::rot_i8i4(
                dim,
                16,
                tail,
                vec![scale; 16],
                vec![lo; tail],
                vec![1.0; tail],
                rotation,
            )
        };
        assert!(build(1.0, 8, 0.0, rotation.clone()).is_ok());
        assert!(
            build(1.0, 7, 0.0, vec![0.0; 23 * dim]).is_err(),
            "an odd tail"
        );
        assert!(
            build(1.0, 8, 1.0, rotation.clone()).is_err(),
            "an empty range"
        );
        assert!(
            build(0.0, 8, 0.0, rotation.clone()).is_err(),
            "a zero scale"
        );
        let mut nan = rotation;
        nan[3] = f32::NAN;
        assert!(build(1.0, 8, 0.0, nan).is_err());
        assert!(CodecSpec::parse(ROT_I8I4_NAME, 1.0)
            .unwrap_err()
            .contains("--rotation"));

        let rotated = codec.rotated().unwrap();
        let (mut codes, mut twice) = (vec![0u8; 20], vec![0u8; 20]);
        let (mut y, mut decoded, mut projected) =
            (vec![0f64; 24], vec![0f32; dim], vec![0f32; dim]);
        for vector in &vectors[..50] {
            let encoded = codec.encode(vector, &mut codes);
            codec.encode(vector, &mut twice);
            assert_eq!(codes, twice);
            assert_eq!(
                encoded.clipped, 0,
                "fitted on these vectors: inside every range"
            );
            rotate(&rotated.rotation, vector, &mut y);
            for (j, byte) in codes[16..].iter().enumerate() {
                for (half, code) in [byte & 0x0F, byte >> 4].into_iter().enumerate() {
                    let d = 2 * j + half;
                    let value = y[16 + d] as f32;
                    let expected =
                        ((value - rotated.lo[d]) / (rotated.hi[d] - rotated.lo[d]) * 15.0).round();
                    assert_eq!(f32::from(code), expected);
                }
            }
            codec.decode(&codes, encoded.scale, &mut decoded);
            unrotate(&rotated.rotation, &y, &mut projected);
            let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
            for (a, b) in decoded.iter().zip(&projected) {
                ab += f64::from(a * b);
                aa += f64::from(a * a);
                bb += f64::from(b * b);
            }
            assert!(ab / (aa.sqrt() * bb.sqrt()) > 0.99);
        }
        let loud: Vec<f32> = vectors[0].iter().map(|x| x * 50.0).collect();
        assert!(codec.encode(&loud, &mut codes).clipped > 0);
    }
}
