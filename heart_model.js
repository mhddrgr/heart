/*
 * Browser version of the heart disease ensemble (no Python needed).
 * Reproduces predict_proba() from train_heart.py using model.json, which is
 * written by export_web_model.py. The ensemble is the equal-weight average of:
 *   logistic regression, RBF SVM (Platt scaling), random forest,
 *   histogram gradient boosting, and a 5-seed neural network.
 */
(function (root) {
  "use strict";

  // Feature preprocessing: standardize numeric columns, one-hot encode categoricals.
  function preprocess(model, row) {
    const z = [];
    model.numeric.forEach((name, i) => {
      z.push((Number(row[name]) - model.scaler_mean[i]) / model.scaler_scale[i]);
    });
    model.categorical.forEach((name) => {
      const value = Number(row[name]);
      model.categories[name].forEach((c) => z.push(value === c ? 1 : 0));
    });
    return z;
  }

  function sigmoid(x) {
    return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
  }

  function logregProba(m, z) {
    let s = m.b;
    for (let i = 0; i < z.length; i++) s += m.coef[i] * z[i];
    return sigmoid(s);
  }

  // libsvm Platt scaling: P = 1 / (1 + exp(A*f + B)), written in a stable form.
  function svmProba(m, z) {
    let f = m.b;
    for (let i = 0; i < m.sv.length; i++) {
      let d2 = 0;
      const sv = m.sv[i];
      for (let j = 0; j < z.length; j++) {
        const diff = z[j] - sv[j];
        d2 += diff * diff;
      }
      f += m.dual[i] * Math.exp(-m.gamma * d2);
    }
    const fApB = m.A * f + m.B;
    return fApB >= 0 ? Math.exp(-fApB) / (1 + Math.exp(-fApB)) : 1 / (1 + Math.exp(fApB));
  }

  function treeLeafValue(t, z, getNext) {
    let node = 0;
    while (t.l[node] !== -1) node = getNext(t, node, z);
    return node;
  }

  function forestProba(m, z) {
    let total = 0;
    m.trees.forEach((t) => {
      // scikit-learn trees compare float32 inputs, so round the same way.
      const leaf = treeLeafValue(t, z, (tree, node, x) =>
        Math.fround(x[tree.f[node]]) <= tree.t[node] ? tree.l[node] : tree.r[node]);
      total += t.p[leaf];
    });
    return total / m.trees.length;
  }

  function histGbProba(m, z) {
    let raw = m.baseline;
    m.trees.forEach((t) => {
      let node = 0;
      while (!t.leaf[node]) {
        const x = z[t.f[node]];
        let goLeft;
        if (Number.isNaN(x)) goLeft = t.ml[node] === 1;
        else goLeft = x <= t.t[node];
        node = goLeft ? t.l[node] : t.r[node];
      }
      raw += t.v[node];
    });
    return sigmoid(raw);
  }

  function mlpSeedProba(layers, z) {
    let h = z;
    layers.forEach((layer, idx) => {
      const out = layer.b.map((bias, o) => {
        let s = bias;
        const row = layer.W[o];
        for (let i = 0; i < h.length; i++) s += row[i] * h[i];
        return s;
      });
      h = idx < layers.length - 1 ? out.map((v) => (v > 0 ? v : 0)) : out;
    });
    return sigmoid(h[0]);
  }

  function mlpProba(m, z) {
    const probs = m.seeds.map((layers) => mlpSeedProba(layers, z));
    return probs.reduce((a, b) => a + b, 0) / probs.length;
  }

  // Returns probabilities of heart disease for one patient.
  function predict(model, row) {
    const z = preprocess(model, row);
    const out = {
      logreg: logregProba(model.logreg, z),
      svm: svmProba(model.svm, z),
      random_forest: forestProba(model.random_forest, z),
      hist_gb: histGbProba(model.hist_gb, z),
      mlp: mlpProba(model.mlp, z),
    };
    const values = Object.values(out);
    out.ensemble = values.reduce((a, b) => a + b, 0) / values.length;
    return out;
  }

  const api = { predict, preprocess };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.HeartModel = api;
})(typeof self !== "undefined" ? self : this);
