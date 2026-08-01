
// Rust requires a massive song-and-dance

type Vector<const N: usize> = [f64; N];
type Matrix<const N: usize> = [[f64; N]; N];

pub fn dot<const N: usize>(x: &Vector<N>, y: &Vector<N>) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
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

pub fn mul<const N: usize>(m: &Matrix<N>, v: &Vector<N>) -> Vector<N> {
    std::array::from_fn(|i| dot(&m[i], v))
}

pub fn add<const N: usize>(x: &Vector<N>, y: &Vector<N>) -> Vector<N> {
    std::array::from_fn(|i| x[i] + y[i])
}

pub fn sub<const N: usize>(x: &Vector<N>, y: &Vector<N>) -> Vector<N> {
    std::array::from_fn(|i| x[i] - y[i])
}