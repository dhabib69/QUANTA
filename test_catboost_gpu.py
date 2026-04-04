from catboost import CatBoostClassifier
import numpy as np

try:
    model = CatBoostClassifier(iterations=10, task_type="GPU", devices='0')
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y, verbose=False)
    print("✅ CatBoost successfully trained on GPU!")
except Exception as e:
    print(f"❌ CatBoost GPU training failed: {e}")
