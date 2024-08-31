mod layerdata;

use pyo3::prelude::*;
use std::collections::HashMap;
use std::vec::Vec;
use layerdata::{LayerData, ArrayBaseRawD}; //Foo, , ArrayBaseRawD
use once_cell::sync::Lazy;
use std::sync::Mutex;
use itertools::iproduct;
//use itertools::Itertools::iproduct;
//use nalgebra::Matrix3;
//use ndarray::IxDyn;
use numpy::ndarray::prelude::s;
use numpy::ndarray::{ArrayD, IxDyn}; //RawArrayView, , ArrayBase, RawViewRepr
use numpy::{PyReadonlyArrayDyn, PyArrayMethods}; //IntoPyArray, PyArrayDyn, 

pub const CONST_OK: &str = "OK";
pub const CONST_KEY_NOT_FOUND: &str = "KEY_NOT_FOUND";
pub const CONST_LAYERS_NOT_FOUND: &str = "LAYERS_NOT_FOUND";
//static mut RAW_DATA: Lazy<Mutex<HashMap<usize, Vec<Box<dyn Foo>>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static mut RAW_DATA: Lazy<Mutex<HashMap<usize, Vec<LayerData>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[pyfunction]
#[pyo3(name = "add_builder_code")]
pub fn add_builder_code_py() -> PyResult<usize> {
    Ok(add_builder_code())
}

#[pyfunction]
#[pyo3(name = "add_layer_data")]
pub fn add_layer_data_py<'py>(builder_code: usize, lweight: f64, lbias: f64
    , dweight: PyReadonlyArrayDyn<'py, f64>, dbias: PyReadonlyArrayDyn<'py, f64>) -> PyResult<String> {
    //let dweight_raw = dweight.as_raw_array().into_dyn();
    //let dbias_raw = dbias.as_raw_array().into_dyn();
    //let mut xx = ArrayD::<f64>::zeros(IxDyn(&[5, 6, 3, 4]));
    //xx.try .clone_from(dweight);
    //let xx = dweight.to_owned_array(); //.into_dyn();
    //let xx = dweight.as_raw_array().into_dyn().into_owned();
    let layer_data = LayerData::new(lweight, lbias, dweight.to_owned_array() /*.as_raw_array().into_dyn()*/, dbias.to_owned_array()/*.as_raw_array().into_dyn()*/);
    //let slice = dbias.into_dy //.readonly().as_array() as DMatrix;
    //let xxx = Matrix3::new(30, 36, 42, 66, 81, 96, 102, 126, 150);
    Ok(add_layer_data(builder_code, layer_data).to_string())
}

#[pyfunction]
#[pyo3(name = "remove_builder_code")]
pub fn remove_builder_code_py(builder_code: usize) -> PyResult<String> {
    Ok(remove_builder_code(builder_code).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn ntk_ffn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_builder_code_py, m)?)?;
    m.add_function(wrap_pyfunction!(add_layer_data_py, m)?)?;
    m.add_function(wrap_pyfunction!(remove_builder_code_py, m)?)?;
    Ok(())
}

fn add_builder_code() -> usize {
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

fn remove_builder_code(builder_code: usize) -> &'static str {
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

fn add_layer_data(builder_code: usize, layer_data: LayerData) -> &'static str {

    //let value = dweight.to //ArrayD::<f64>::from(dweight.);
    //dweight.
    let mut binding = unsafe {RAW_DATA.lock().unwrap()};
    let result = binding.get_mut(&builder_code);
    if result.is_none() {
        return CONST_KEY_NOT_FOUND
    } else {
        let vector = result.unwrap();
        //let layer_data = Box::new(LayerData::new(lweight, lbias, dweight, dbias)) as Box<dyn Foo>;
        //let layer_data = LayerData::new(lweight, lbias, dweight, dbias);
        vector.push(layer_data);
        return CONST_OK
    }
}

fn get_ntk(builder_code: usize) -> Result<ArrayD<f64>, &'static str> {
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
    for (kk1, kk2, alpha1, alpha2) in iproduct!(0..output_dim, 0..output_dim, 0..batch_size, 0..batch_size) {
        if ntk[[kk1, kk2, alpha1, alpha2]] == 0.0 {
            let mut value: f64 = 0.0;
            for layer in layer_list {
                value += (&layer.dbias.slice(s![kk1, alpha1, ..]) * &layer.dbias.slice(s![kk2, alpha2, ..])).sum() * layer.lbias;
                value += (&layer.dweight.slice(s![kk1, alpha1, ..]) * &layer.dweight.slice(s![kk2, alpha2, ..])).sum() * layer.lweight;
                //value += tensor_dot(kk1, kk2, alpha1, alpha2, layer.dbias, layer.lbias);
                //value += tensor_dot(kk1, kk2, alpha1, alpha2, layer.dweight, layer.lweight);
            }
            ntk[[kk1, kk2, alpha1, alpha2]] = value;
            ntk[[kk2, kk1, alpha2, alpha1]] = value;
        }
    }
    return Ok(ntk);//ArrayD::<f64>::ones(IxDyn(&[5, 6, 3, 4]));
}

/*fn tensor_dot(kk1: usize, kk2: usize, alpha1: usize, alpha2: usize, data: ArrayBaseRawD<f64>, multiplier: f64)  -> f64 {
    let shape = data.shape();
    let mut value: f64 = 0.0;
    //let iter1 = data.axis_iter(IxDyn(&[kk1, alpha1])); //.lanes(axis)
    if shape.len() == 3 {
        for idx2 in 0.. shape[2] {
            value += data[[kk1, alpha1, idx2]] * data[[kk2, alpha2, idx2]]
        }
    }
    if shape.len() == 4 {
        for idx2 in 0.. shape[2] {
            for idx3 in 0.. shape[3] {
                value += data[[kk1, alpha1, idx2, idx3]] * data[[kk2, alpha2, idx2, idx3]]
            }
        }
    }
    return value * multiplier;
}*/
/*
use ndarray::ArrayD;
use ndarray::IxDyn;

// Create a 5 × 6 × 3 × 4 array using the dynamic dimension type
let mut a = ArrayD::<f64>::zeros(IxDyn(&[5, 6, 3, 4]));
*/