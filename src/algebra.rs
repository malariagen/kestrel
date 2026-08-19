pub type Vector<const N: usize> = [f64; N];
pub type Matrix<const N: usize> = [[f64; N]; N];

pub fn dot<const N: usize>(x: &Vector<N>, y: &Vector<N>) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        sum = x[i].mul_add(y[i], sum);
    }

    sum
}

pub fn outer<const N: usize>(x: &Vector<N>, y: &Vector<N>) -> Matrix<N> {
    let mut m = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            m[i][j] = x[i] * y[j];
        }
    }
    m
}

pub fn dot_n<const N: usize>(n: usize, x: &Vector<N>, y: &Vector<N>) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        sum = x[i].mul_add(y[i], sum);
    }
    sum
}

pub fn sum<const N: usize>(x: &Vector<N>) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        sum += x[i];
    }
    sum
}

pub fn sum_n<const N: usize>(n: usize, x: &Vector<N>) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        sum += x[i];
    }
    sum
}

pub fn mul<const N: usize>(m: &Matrix<N>, v: &Vector<N>) -> Vector<N> {
    std::array::from_fn(|i| dot(&m[i], v))
}

pub fn mul_n<const N: usize>(n: usize, m: &Matrix<N>, v: &Vector<N>) -> Vector<N> {
    let mut p = [0.0; N];
    for i in 0..n {
        p[i] = dot_n(n, &m[i], v);
    }
    p
}

pub fn scale_mul<const N: usize>(a: f64, x: &Vector<N>) -> Vector<N> {
    std::array::from_fn(|i| a * x[i])
}

pub fn scale_div<const N: usize>(d: f64, x: &Vector<N>) -> Vector<N> {
    std::array::from_fn(|i| x[i] / d)
}

pub fn add<const N: usize>(x: &Vector<N>, y: &Vector<N>) -> Vector<N> {
    std::array::from_fn(|i| x[i] + y[i])
}

pub fn add_n<const N: usize>(n: usize, x: &Vector<N>, y: &Vector<N>) -> Vector<N> {
    let mut s = [0.0; N];
    for i in 0..n {
        s[i] = x[i] + y[i];
    }
    s
}

pub fn sub<const N: usize>(x: &Vector<N>, y: &Vector<N>) -> Vector<N> {
    std::array::from_fn(|i| x[i] - y[i])
}

pub fn sum_to_one<const N: usize>(x: &Vector<N>) -> Vector<N> {
    let s = sum(x);
    std::array::from_fn(|i| x[i] / s)
}

pub fn scale_div_mut<const N: usize>(m: &mut Matrix<N>, a: f64) {
    for i in 0..N {
        for j in 0..N {
            m[i][j] /= a;
        }
    }
}
