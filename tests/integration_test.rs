use ntk_ffn;

#[test]
fn it_add_builder_code() {
    assert_eq!(1, ntk_ffn::add_builder_code_py().unwrap());
}

#[test]
fn it_remove_builder_code() {
    assert_eq!(ntk_ffn::CONST_KEY_NOT_FOUND.to_string(), ntk_ffn::remove_builder_code_py(146).unwrap());
}

//dweight: PyReadonlyArrayDyn<'py, f64> -> ArrayD<f64>