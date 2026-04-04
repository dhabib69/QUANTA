import numpy as np

try:
    features_scaled = np.random.randn(1, 214)
    seed = 42
    n_features = features_scaled.shape[1]
    n_keep = int(n_features * 0.7)
    rng = np.random.RandomState(seed)
    feature_indices = np.sort(rng.choice(n_features, n_keep, replace=False))
    
    # 1. Does slicing throw?
    f_masked = features_scaled[:, feature_indices]
    print("Mask slicing OK!")
    
    # 2. Could preds_array throw?
    preds_list = [np.random.rand(1, 2), np.random.rand(1, 2)]
    preds_array = np.array(preds_list)
    print("preds_array shape:", preds_array.shape)
    x = preds_array[:, :, 1].T
    print("Array subset OK!")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAILED:", e)
