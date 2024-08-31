//use numpy::ndarray::RawArrayView;
use numpy::ndarray::{IxDyn, ArrayBase, ViewRepr, ArrayD, RawViewRepr}; //RawArrayView, ArrayD, 

pub type ArrayBaseRawD<A> = ArrayBase<RawViewRepr<*const A>, IxDyn>;
//pub type ArrayBaseRawD<A> = ArrayBase<ViewRepr<&A>, IxDyn>;

/*pub trait Foo {
    fn value(&self) -> f64;
}*/

pub struct LayerData {
    pub lweight: f64,
    pub lbias: f64,
    pub dweight: ArrayD<f64>,//ArrayBaseRawD<f64>,
    pub dbias: ArrayD<f64> //ArrayBaseRawD<f64>
}
/*pub struct LayerData {
    pub lweight: f64,
    pub lbias: f64,
    pub dweight: ArrayViewD<f64>,
    pub dbias: ArrayViewD<f64>
}*/

impl LayerData {
    pub fn new(lweight: f64, lbias: f64, dweight: ArrayD<f64>/*ArrayBaseRawD<f64>*/, dbias:ArrayD<f64> /*ArrayBaseRawD<f64>*/) -> LayerData {
        LayerData { lweight, lbias, dweight, dbias }
    }
}

/*impl <D, E>Foo for LayerData<D, E> {
    fn value(&self) -> f64 {
        self.lbias
    }
}*/

/*impl LayerData<'_> {
    pub fn new<'a>(lweight: f64, lbias: f64, dweight: ArrayViewD<'a, f64>, dbias: ArrayViewD<'a, f64>) -> LayerData<'a> {
        LayerData { lweight, lbias, dweight, dbias }
    }
}*/
