use ntk_ffn;
use ntk_ffn::layerdata::LayerData;
use numpy::ndarray::prelude::*;

#[test]
fn it_add_builder_code() {
    assert_eq!(1, ntk_ffn::add_builder_code());
}

#[test]
fn it_remove_builder_code() {
    assert_eq!(ntk_ffn::CONST_KEY_NOT_FOUND.to_string(), ntk_ffn::remove_builder_code(146));
}

#[test]
fn it_get_ntk() {
    let dweight = ArrayD::<f64>::ones(IxDyn(&[26, 64, 250, 28*28]));//])); //
     //array![[1.,2.,3.], [4.,5.,6.]].to_owned();
    let dbias = ArrayD::<f64>::ones(IxDyn(&[26, 64, 250])); //26, 64
    let builder_code = ntk_ffn::add_builder_code();
    let layer_data = LayerData::new(0.01, 0.025, dweight, dbias);
    ntk_ffn::add_layer_data(builder_code, layer_data);
    let ntk = ntk_ffn::get_ntk(builder_code).unwrap();
    let shape = ntk.shape();
    print!("{}-{}-{}-{}", shape[0], shape[1], shape[2], shape[3]);
    assert_eq!(ntk_ffn::CONST_OK.to_string(), ntk_ffn::remove_builder_code(builder_code));
}
