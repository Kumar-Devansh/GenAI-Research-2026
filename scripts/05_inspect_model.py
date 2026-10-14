"""

This script loads the trained model (.pkl file) and displays
information about it. This allows you to "look inside" the model
and understand how it works.
"""

from common import MODEL_DIR
import joblib

# Load the model
model_path = MODEL_DIR / "gradient_boosting.pkl"
if not model_path.is_file():
    raise FileNotFoundError(f"Trained model not found: {model_path}. Run scripts/04_train_ml_models.py first.")
model = joblib.load(model_path)

print("Model type:", type(model))
print("\nModel parameters:")
print(model.get_params())

# If it is a tree-based model, you can view the feature importance.
if hasattr(model, "feature_importances_"):
    print("\nFeature importances (top 10):")
    import numpy as np
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for i in top_idx:
        print(f"  Feature {i}: {importances[i]:.4f}")
