//! The rotation `rot192-i8i4` stores vectors under, and how it is fitted.
//!
//! Every step is a fixed sequence of IEEE operations — sums in index order, `sqrt` and
//! division only — so the same vectors fit the same bytes on every machine, and a vector
//! rotates to the same bytes wherever it is encoded.

/// `y = R · x` in `f64`, `R` being `out.len()` rows of `x.len()` components, row-major.
pub(crate) fn rotate(rotation: &[f32], x: &[f32], out: &mut [f64]) {
    let dim = x.len();
    debug_assert_eq!(rotation.len(), out.len() * dim);
    for (value, row) in out.iter_mut().zip(rotation.chunks_exact(dim)) {
        let mut sum = 0f64;
        for (r, x) in row.iter().zip(x) {
            sum += f64::from(*r) * f64::from(*x);
        }
        *value = sum;
    }
}

/// `x = Rᵀ · y`, back in the input space, each component summed over the rows in order.
pub(crate) fn unrotate(rotation: &[f32], y: &[f64], out: &mut [f32]) {
    let dim = out.len();
    debug_assert_eq!(rotation.len(), y.len() * dim);
    for (d, value) in out.iter_mut().enumerate() {
        let mut sum = 0f64;
        for (row, y) in rotation.chunks_exact(dim).zip(y) {
            sum += f64::from(row[d]) * y;
        }
        *value = sum as f32;
    }
}

/// The uncentered second moment `E[x xᵀ]` of `vectors`, `dim × dim` row-major, every
/// entry summed over the vectors in order.
pub(crate) fn second_moment<'a>(
    vectors: impl IntoIterator<Item = &'a [f32]>,
    dim: usize,
) -> Vec<f64> {
    let mut sums = vec![0f64; dim * dim];
    let mut x = vec![0f64; dim];
    let mut n = 0u64;
    for vector in vectors {
        for (to, from) in x.iter_mut().zip(vector) {
            *to = f64::from(*from);
        }
        for i in 0..dim {
            let xi = x[i];
            let row = &mut sums[i * dim..(i + 1) * dim];
            for j in i..dim {
                row[j] += xi * x[j];
            }
        }
        n += 1;
    }
    let n = n.max(1) as f64;
    for i in 0..dim {
        for j in i..dim {
            let value = sums[i * dim + j] / n;
            sums[i * dim + j] = value;
            sums[j * dim + i] = value;
        }
    }
    sums
}

/// The `rows` leading eigenvectors of the symmetric `dim × dim` matrix `m`, as `f32` rows:
/// cyclic Jacobi, eigenvalues in decreasing order (the lower index first on a tie), each
/// vector's largest component — the first, on a tie — made positive.
pub(crate) fn principal_rows(mut m: Vec<f64>, dim: usize, rows: usize) -> Vec<f32> {
    assert_eq!(m.len(), dim * dim);
    assert!(rows <= dim);
    let mut v = vec![0f64; dim * dim];
    for i in 0..dim {
        v[i * dim + i] = 1.0;
    }
    for sweep in 0..100 {
        let off: f64 = (0..dim)
            .flat_map(|p| (p + 1..dim).map(move |q| (p, q)))
            .map(|(p, q)| m[p * dim + q].abs())
            .sum();
        if off == 0.0 {
            break;
        }
        for p in 0..dim {
            for q in p + 1..dim {
                let apq = m[p * dim + q];
                let (app, aqq) = (m[p * dim + p], m[q * dim + q]);
                // A rotation too small to change either diagonal entry is dropped.
                if sweep > 3
                    && (app.abs() + apq.abs() * 1e2 == app.abs())
                    && (aqq.abs() + apq.abs() * 1e2 == aqq.abs())
                {
                    m[p * dim + q] = 0.0;
                    m[q * dim + p] = 0.0;
                    continue;
                }
                if apq == 0.0 {
                    continue;
                }
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta.abs() > 1e150 {
                    0.5 / theta
                } else {
                    let t = 1.0 / (theta.abs() + (theta * theta + 1.0).sqrt());
                    if theta < 0.0 {
                        -t
                    } else {
                        t
                    }
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);
                m[p * dim + p] = app - t * apq;
                m[q * dim + q] = aqq + t * apq;
                m[p * dim + q] = 0.0;
                m[q * dim + p] = 0.0;
                for r in 0..dim {
                    if r != p && r != q {
                        let (arp, arq) = (m[r * dim + p], m[r * dim + q]);
                        let new_p = arp - s * (arq + tau * arp);
                        let new_q = arq + s * (arp - tau * arq);
                        m[r * dim + p] = new_p;
                        m[p * dim + r] = new_p;
                        m[r * dim + q] = new_q;
                        m[q * dim + r] = new_q;
                    }
                    let (vrp, vrq) = (v[r * dim + p], v[r * dim + q]);
                    v[r * dim + p] = vrp - s * (vrq + tau * vrp);
                    v[r * dim + q] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|a, b| m[b * dim + b].total_cmp(&m[a * dim + a]).then(a.cmp(b)));
    let mut out = Vec::with_capacity(rows * dim);
    for &column in order.iter().take(rows) {
        let component = |r: usize| v[r * dim + column];
        let mut largest = 0;
        for r in 1..dim {
            if component(r).abs() > component(largest).abs() {
                largest = r;
            }
        }
        let sign = if component(largest) < 0.0 { -1.0 } else { 1.0 };
        out.extend((0..dim).map(|r| (sign * component(r)) as f32));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::oxv::testing::Random;

    /// A diagonal matrix's eigenvectors are the axes, in decreasing order of their entries,
    /// with the lower index first on a tie.
    #[test]
    fn a_diagonal_matrix_gives_the_axes_in_order() {
        let dim = 4;
        let mut m = vec![0f64; 16];
        for (i, value) in [1.0, 3.0, 2.0, 3.0].into_iter().enumerate() {
            m[i * dim + i] = value;
        }
        let rows = principal_rows(m, dim, 3);
        assert_eq!(
            rows,
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        );
    }

    /// On a random symmetric matrix the rows are orthonormal eigenvectors, by decreasing
    /// eigenvalue, each with its largest component positive.
    #[test]
    fn the_rows_are_orthonormal_eigenvectors() {
        let dim = 24;
        let mut random = Random(31);
        let vectors: Vec<Vec<f32>> = (0..300).map(|_| random.unit(dim)).collect();
        let m = second_moment(vectors.iter().map(Vec::as_slice), dim);
        let rows = principal_rows(m.clone(), dim, dim);
        let mut previous = f64::INFINITY;
        for (i, row) in rows.chunks_exact(dim).enumerate() {
            for (j, other) in rows.chunks_exact(dim).enumerate() {
                let dot: f64 = row.iter().zip(other).map(|(a, b)| f64::from(a * b)).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-5, "{i} {j} {dot}");
            }
            // M · r = λ r, with λ = rᵀ M r.
            let mr: Vec<f64> = (0..dim)
                .map(|a| (0..dim).map(|b| m[a * dim + b] * f64::from(row[b])).sum())
                .collect();
            let lambda: f64 = mr.iter().zip(row).map(|(a, b)| a * f64::from(*b)).sum();
            for (a, b) in mr.iter().zip(row) {
                assert!((a - lambda * f64::from(*b)).abs() < 1e-6);
            }
            assert!(lambda <= previous + 1e-12);
            previous = lambda;
            let largest =
                row.iter()
                    .copied()
                    .fold(0f32, |best, x| if x.abs() > best.abs() { x } else { best });
            assert!(largest > 0.0);
        }
    }
}
