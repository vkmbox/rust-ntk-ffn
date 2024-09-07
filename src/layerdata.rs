use numpy::ndarray::ArrayD;

pub struct LayerData {
    pub lweight: f64,
    pub lbias: f64,
    pub dweight: ArrayD<f64>,
    pub dbias: ArrayD<f64>
}

impl LayerData {
    pub fn new(lweight: f64, lbias: f64, dweight: ArrayD<f64>, dbias: ArrayD<f64>) -> LayerData {
        LayerData { lweight, lbias, dweight, dbias }
    }
}
