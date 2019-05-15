pub static PERCEPTRON_PATH : & str = "perceptron.yaml";

pub fn usage<'a>() -> &'a str {
"Usage: ./multilayer_perceptron [-a] [args]+

-a[nalytics]\t[/path/to/dataset.csv]\t\trun analytic module.
-l[earning]\t[/path/to/dataset.csv] [args]*\trun learning module, optional arguments are `-width [width]` and `-height [height]`, respectively the width and height of the perceptron hidden layers.
-p[redict]\t[/path/to/dataset.csv]\t\trun prediction module.
"
}