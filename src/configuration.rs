pub static PERCEPTRON_PATH : & str = "perceptron.yaml";

pub fn usage<'a>() -> &'a str {
"Usage: ./multilayer_perceptron [-a] [args]+

-a[nalytics]\t[/path/to/dataset.csv]\t\trun analytic module.
-l[earning]\t[/path/to/dataset.csv]\t\trun learning module.
-p[redict]\t[/path/to/dataset.csv]\t\trun prediction module."
}