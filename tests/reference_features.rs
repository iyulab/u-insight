//! ANOVA F-tests, mutual information and permutation importance against SciPy and
//! the published definitions, on a seeded three-class draw (n = 45, four features:
//! two informative, one weak, one noise). The constants were printed by that
//! recipe with the versions in the next line.
#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop
)]

// Generated with NumPy 2.4.6, SciPy 1.17.1 (seed 20261023, n = 45, 4 features, 3 classes).
const NF: usize = 4;
const FEATURES: [[f64; 45]; 4] = [
    [
        1.687035, 1.954512, 1.173747, -1.637572, 0.326462, -0.088668, 0.741741, 0.383878,
        -1.084034, 0.397992, 0.016716, 0.249418, 1.007232, 0.824605, 0.264937, 0.540741, 2.217636,
        3.202808, 0.487986, 2.833639, 3.799843, 1.408668, 1.229078, 1.701014, -0.252911, 4.421515,
        1.00069, 2.485402, 2.594782, 1.675248, 3.552811, 4.722875, 3.698738, 3.137729, 5.942426,
        3.138118, 5.289742, 5.521609, 4.115864, 3.950631, 4.503901, 4.792252, 3.951819, 3.537625,
        2.466764,
    ],
    [
        -0.88883, 0.153088, 2.676187, 1.192865, -1.227964, 0.2648, -0.096898, 1.613955, 0.226767,
        0.996491, 0.564998, -0.468354, 2.208412, -0.398045, 0.568769, 0.687936, 1.08715, 0.298517,
        -0.209751, -0.195292, 0.006251, 1.392389, 0.861276, -0.933223, -0.127771, 1.397778,
        -0.254257, 0.652374, -0.568314, 0.187493, -0.212108, -0.317427, 3.835574, 0.310354,
        0.838105, 1.206042, 0.616447, 2.319645, 2.791532, -0.626778, -0.089596, -0.862531,
        1.813487, 0.880901, 0.305187,
    ],
    [
        0.817058, 0.909191, -0.949302, 1.671416, -0.09599, -1.068115, -0.191832, -0.822109,
        -3.149982, 1.438059, 0.978093, 0.09477, 1.807781, 1.706854, -0.131775, -0.585484, 0.836752,
        -0.253805, 1.920352, 1.25329, -0.309313, 0.114021, 1.63442, -0.775089, -0.939262, 1.69131,
        0.336977, -0.639811, -1.079987, 0.579387, 1.685659, -1.608873, -0.096088, -0.958067,
        1.222437, 0.806928, 0.188509, 0.048402, -0.056502, 1.367827, -0.181878, 1.486614, 0.028013,
        -0.37793, -0.295372,
    ],
    [
        8.501954, 7.053358, 10.154143, 1.175795, 7.721656, -1.012592, 5.691419, 2.496055, 1.603864,
        7.419302, 6.237195, 6.486029, 3.650761, 6.81028, 4.327238, 4.236015, 1.086171, 0.990325,
        -1.524343, 4.272208, 2.318734, 1.760929, 0.200121, 2.009181, 5.387917, 1.572908, 6.153871,
        2.673558, 5.05941, 5.949072, 1.58706, -1.270535, -0.006904, 3.808758, 0.870014, 3.870756,
        -1.556137, 5.533977, 1.119314, -0.745874, 3.663242, 3.058205, 0.527484, 0.850358, 2.827966,
    ],
];
const CLASSES: [usize; 45] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
];
const ANOVA_F: [f64; 4] = [
    45.65491765544417,
    1.0663557723718586,
    0.008267772439780281,
    7.942030362495669,
];
const ANOVA_P: [f64; 4] = [
    2.925021538770972e-11,
    0.35339650834486463,
    0.9917679252955952,
    0.0011871245836475432,
];
const MI: [f64; 4] = [
    0.517765104187947,
    0.12890780305919072,
    0.11728006653108965,
    0.298042646454325,
];
const Y: [f64; 45] = [
    -3.295282, -3.384051, -6.082211, -3.91584, -6.284318, 0.201705, -2.726573, -0.72239, -2.721401,
    -5.406635, -4.824595, -5.783719, -0.609622, -3.619157, -1.941437, -1.866279, 2.191257,
    5.124107, 1.325616, 1.117831, 3.778067, 1.293406, 0.558888, 0.264384, -4.831355, 4.374956,
    -4.309619, 1.775903, -0.74114, -2.664904, 3.740032, 8.202246, 5.41013, 0.755257, 8.198788,
    1.971919, 9.027522, 3.495022, 5.077176, 6.226751, 3.578987, 4.535008, 5.508639, 4.597815,
    1.061461,
];

use u_insight::analysis::{anova_feature_selection, mutual_info_classif};
use u_insight::feature_importance::permutation_importance;

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let scale = want.abs().max(1e-300);
    assert!(
        (got - want).abs() <= rel * scale,
        "{what}: got {got}, want {want}"
    );
}

fn features() -> (Vec<Vec<f64>>, Vec<String>) {
    (
        FEATURES.iter().map(|c| c.to_vec()).collect(),
        (0..NF).map(|j| format!("f{j}")).collect(),
    )
}

#[test]
fn anova_f_and_p_match_scipy_f_oneway() {
    let (cols, names) = features();
    let r = anova_feature_selection(&cols, &names, &CLASSES, 0.05).unwrap();
    for j in 0..NF {
        let f = r
            .features
            .iter()
            .find(|f| f.name == names[j])
            .unwrap_or_else(|| panic!("feature {j} missing"));
        close(f.f_statistic, ANOVA_F[j], 1e-9, &format!("F {j}"));
        close(f.p_value, ANOVA_P[j], 1e-6, &format!("p {j}"));
    }
    assert_eq!(
        r.selected_indices
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        [0usize, 3].into()
    );
}

/// Equal-frequency bins by rank (Sturges' count), plug-in mutual information
/// in nats -- restated from the definition.
#[test]
fn mutual_information_matches_the_plug_in_definition() {
    let (cols, names) = features();
    let r = mutual_info_classif(&cols, &names, &CLASSES, None).unwrap();
    for j in 0..NF {
        let f = r.features.iter().find(|f| f.name == names[j]).unwrap();
        close(f.mi, MI[j], 1e-9, &format!("MI {j}"));
    }
}

/// A shuffle-based score depends on the random stream, so only the structure
/// is compared: the two features that generate the target matter, the other
/// two do not, in any unit of the target.
#[test]
fn permutation_importance_ranks_the_generating_features_first() {
    let (cols, names) = features();
    for factor in [1.0, 1e-9, 1e6] {
        let y: Vec<f64> = Y.iter().map(|v| v * factor).collect();
        let r = permutation_importance(&cols, &names, &y, 20, 7).unwrap();
        let imp = |j: usize| {
            r.features
                .iter()
                .find(|f| f.name == names[j])
                .unwrap()
                .importance
        };
        assert!(
            imp(0) > 0.3 && imp(3) > 0.05,
            "x{factor:e}: {} {}",
            imp(0),
            imp(3)
        );
        assert!(
            imp(1) < 0.05 && imp(2) < 0.05,
            "x{factor:e}: {} {}",
            imp(1),
            imp(2)
        );
        assert!(
            r.baseline_score > 0.9,
            "x{factor:e}: R2 {}",
            r.baseline_score
        );
    }
}
