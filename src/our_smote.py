"""Base class and original SMOTE methods for over-sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT

import math
import warnings
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import (
    _get_column_indices,
    _safe_indexing,
    check_array,
    check_random_state,
)
from sklearn.utils.sparsefuncs_fast import (
    csr_mean_variance_axis0,
)
from sklearn.utils.validation import _num_features

from imblearn.utils import check_target_type
from imblearn.utils._docstring import *
from imblearn.utils._param_validation import HasMethods, StrOptions
from imblearn.utils._validation import _check_X
from imblearn.utils.fixes import _is_pandas_df
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.over_sampling.base import BaseOverSampler

_random_state_docstring = """random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.
    """.rstrip()

_n_jobs_docstring = """n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.
    """.rstrip()

@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SMOTENC_GENERATIVE(SMOTENC):
    def __init__(
        self,
        categorical_features,
        *,
        categorical_encoder=None,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(
            categorical_features=categorical_features,
            categorical_encoder=categorical_encoder,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.sampling_strategy_ = sampling_strategy

    def fit(self, X):
        self.X = X.copy()
        # FIXME: to be removed in 0.12
        if self.n_jobs is not None:
            warnings.warn(
                "The parameter `n_jobs` has been deprecated in 0.10 and will be "
                "removed in 0.12. You can pass an nearest neighbors estimator where "
                "`n_jobs` is already set instead.",
                FutureWarning,
            )

        self.n_features_ = _num_features(X)
        self.n_features_in_ = self.n_features_
        self.y_size = len(X)

        self._validate_column_types(X)
        self._validate_estimator()

        X_continuous = _safe_indexing(X, self.continuous_features_, axis=1)
        X_continuous = check_array(X_continuous, accept_sparse=["csr", "csc"])
        X_categorical = _safe_indexing(X, self.categorical_features_, axis=1)
        if X_continuous.dtype.name != "object":
            dtype_ohe = X_continuous.dtype
        else:
            dtype_ohe = np.float64

        if self.categorical_encoder is None:
            self.categorical_encoder_ = OneHotEncoder(
                handle_unknown="ignore", dtype=dtype_ohe
            )
        else:
            self.categorical_encoder_ = clone(self.categorical_encoder)

        # the input of the OneHotEncoder needs to be dense
        X_ohe = self.categorical_encoder_.fit_transform(
            X_categorical.toarray() if sparse.issparse(X_categorical) else X_categorical
        )
        if not sparse.issparse(X_ohe):
            X_ohe = sparse.csr_matrix(X_ohe, dtype=dtype_ohe)

        X_encoded = sparse.hstack((X_continuous, X_ohe), format="csr", dtype=dtype_ohe)
        self.X_encoded = X_encoded.copy()

    def generate(self, n_samples):


        
        # SMOTE resampling starts here
        self.median_std_ = {}
        class_sample = 0
        y = np.zeros(self.y_size)
        target_class_indices = np.flatnonzero(y == class_sample)
        X_class = _safe_indexing(self.X_encoded, target_class_indices)

        _, var = csr_mean_variance_axis0(
            X_class[:, : self.continuous_features_.size]
        )
        self.median_std_[class_sample] = np.median(np.sqrt(var))

        # In the edge case where the median of the std is equal to 0, the 1s
        # entries will be also nullified. In this case, we store the original
        # categorical encoding which will be later used for inverting the OHE
        if math.isclose(self.median_std_[class_sample], 0):
            # This variable will be used when generating data
            self._X_categorical_minority_encoded = X_class[
                :, self.continuous_features_.size :
            ].toarray()

        # we can replace the 1 entries of the categorical features with the
        # median of the standard deviation. It will ensure that whenever
        # distance is computed between 2 samples, the difference will be equal
        # to the median of the standard deviation as in the original paper.
        X_class_categorical = X_class[:, self.continuous_features_.size :]
        # With one-hot encoding, the median will be repeated twice. We need
        # to divide by sqrt(2) such that we only have one median value
        # contributing to the Euclidean distance
        X_class_categorical.data[:] = self.median_std_[class_sample] / np.sqrt(2)
        X_class[:, self.continuous_features_.size :] = X_class_categorical

        self.nn_k_.fit(X_class)
        nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
        X_new, y_new = self._make_samples(
            X_class, int, class_sample, X_class, nns, n_samples, 1.0
        )
        X_resampled = sparse.vstack([X_new], format=self.X_encoded.format)
        # SMOTE resampling ends here

        # reverse the encoding of the categorical features
        X_res_cat = X_resampled[:, self.continuous_features_.size :]
        X_res_cat.data = np.ones_like(X_res_cat.data)
        X_res_cat_dec = self.categorical_encoder_.inverse_transform(X_res_cat)

        if sparse.issparse(self.X):
            X_resampled = sparse.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size],
                    X_res_cat_dec,
                ),
                format="csr",
            )
        else:
            X_resampled = np.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size].toarray(),
                    X_res_cat_dec,
                )
            )

        indices_reordered = np.argsort(
            np.hstack((self.continuous_features_, self.categorical_features_))
        )
        if sparse.issparse(X_resampled):
            # the matrix is supposed to be in the CSR format after the stacking
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]

        X_resampled_df = pd.DataFrame(X_resampled, columns=self.X.columns)
        return X_resampled_df