pub mod layerdata;

use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};
use std::collections::HashMap;
use std::vec::Vec;
use layerdata::LayerData;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use itertools::iproduct;
use ndarray_einsum_beta::*;
use numpy::ndarray::prelude::s;
use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};

pub const CONST_OK: &str = "OK";
pub const CONST_KEY_NOT_FOUND: &str = "KEY_NOT_FOUND";
pub const CONST_LAYERS_NOT_FOUND: &str = "LAYERS_NOT_FOUND";
static mut RAW_DATA: Lazy<Mutex<HashMap<usize, Vec<LayerData>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// A Python module implemented in Rust.
#[pymodule]
fn ntk_ffn<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "add_builder_code")]
    pub fn add_builder_code_py() -> PyResult<usize> {
        Ok(add_builder_code())
    }
    
    #[pyfn(m)]
    #[pyo3(name = "add_layer_data")]
    pub fn add_layer_data_py<'py>(builder_code: usize, lweight: f64, lbias: f64
        , dweight: PyReadonlyArrayDyn<'py, f64>, dbias: PyReadonlyArrayDyn<'py, f64>) -> PyResult<String> {
        let layer_data = LayerData::new(lweight, lbias, dweight.to_owned_array() /*.as_raw_array().into_dyn()*/, dbias.to_owned_array()/*.as_raw_array().into_dyn()*/);
        Ok(add_layer_data(builder_code, layer_data).to_string())
    }
    
    #[pyfn(m)]
    #[pyo3(name = "remove_builder_code")]
    pub fn remove_builder_code_py(builder_code: usize) -> PyResult<String> {
        Ok(remove_builder_code(builder_code).to_string())
    }
    
    #[pyfn(m)]
    #[pyo3(name = "get_ntk")]
    pub fn get_ntk_py<'py>(py: Python<'py>, builder_code: usize) -> Bound<'py, PyArrayDyn<f64>> {
        let result = get_ntk(builder_code).unwrap();
        result.into_pyarray_bound(py)
        //Ok(get_ntk(builder_code).to_string())
    }

    Ok(())
}

pub fn add_builder_code() -> usize {
    let mut key: usize = 1;
    let lock = unsafe {RAW_DATA.lock()};
    
    lock.and_then(|mut map| {
        while map.get(&key).is_some() {
            key += 1; //TODO: get as randomly generated
        }
        Ok(map.insert(key, Vec::new()))
    }).map_err(|err| println!("{:?}", err)).ok();
    key
}

pub fn remove_builder_code(builder_code: usize) -> &'static str {
    let lock = unsafe {RAW_DATA.lock()};
    
    lock.and_then(|mut map| {
        return match map.get(&builder_code) {
            None => Ok(CONST_KEY_NOT_FOUND),
            Some(_) => {
                map.remove(&builder_code);
                Ok(CONST_OK)
            }
        }
    }).map_err(|err| println!("{:?}", err)).unwrap()
}

pub fn add_layer_data(builder_code: usize, layer_data: LayerData) -> &'static str {
    let mut binding = unsafe {RAW_DATA.lock().unwrap()};
    let result = binding.get_mut(&builder_code);
    if result.is_none() {
        return CONST_KEY_NOT_FOUND
    } else {
        let vector = result.unwrap();
        vector.push(layer_data);
        return CONST_OK
    }
}

pub fn get_ntk(builder_code: usize) -> Result<ArrayD<f64>, &'static str> {
    let binding = unsafe {RAW_DATA.lock().unwrap()};
    let result = binding.get(&builder_code);
    if result.is_none() {
        return Result::Err(CONST_KEY_NOT_FOUND)
    }
    let layer_list = result.unwrap();
    if layer_list.len() == 0 {
        return Result::Err(CONST_LAYERS_NOT_FOUND)
    }

    let shape = layer_list[0].dbias.shape();
    let output_dim = shape[0]; let batch_size = shape[1];
    let mut ntk = ArrayD::<f64>::zeros(IxDyn(&[output_dim, output_dim, batch_size, batch_size]));
    for layer in layer_list {
        let tmp_bias = einsum("abx,cdx->acbd", &[&layer.dbias, &layer.dbias]).unwrap() * layer.lbias;
        ntk = ntk + tmp_bias;
        let tmp_weight = einsum("abxy,cdxy->acbd", &[&layer.dweight, &layer.dweight]).unwrap() * layer.lweight;
        ntk = ntk + tmp_weight;        
    }
    //SKIPPING calc for kk1 != kk2
    /*for (kk1, kk2, alpha1, alpha2) in iproduct!(*0..output_dim, 0..output_dim, 0..batch_size, 0..batch_size) {
        if ntk[[kk1, kk2, alpha1, alpha2]] == 0.0 {
            let mut value: f64 = 0.0;
            for layer in layer_list {
                value += (&layer.dbias.slice(s![kk1, alpha1, ..]) * &layer.dbias.slice(s![kk2, alpha2, ..])).sum() * layer.lbias;
                value += (&layer.dweight.slice(s![kk1, alpha1, .., ..]) * &layer.dweight.slice(s![kk2, alpha2, .., ..])).sum() * layer.lweight;
            }
            ntk[[kk1, kk2, alpha1, alpha2]] = value;
            ntk[[kk1, kk2, alpha2, alpha1]] = value;
        }
    }*/
    return Ok(ntk);
}

/*
use ndarray::ArrayD;
use ndarray::IxDyn;

// Create a 5 × 6 × 3 × 4 array using the dynamic dimension type
let mut a = ArrayD::<f64>::zeros(IxDyn(&[5, 6, 3, 4]));
*/