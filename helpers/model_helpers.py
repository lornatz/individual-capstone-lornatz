"""
Model Helper Utilities
======================

Simple helper functions for saving and loading machine learning models.

Usage:
    from helpers.model_helpers import save_model, load_model

    # Save a model
    save_model(my_model, 'models/my_model.pkl')

    # Load a model
    loaded_model = load_model('models/my_model.pkl')
"""

import joblib
from pathlib import Path


def save_model(model, filepath):
    """
    Save a model to a pickle file.

    Parameters:
    -----------
    model : object
        The trained model to save (can be any sklearn model, scaler, encoder, etc.)
    filepath : str
        Path where the model will be saved (e.g., 'models/my_model.pkl')

    Returns:
    --------
    str : The path where the model was saved

    Example:
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> model = LinearRegression()
    >>> model.fit(X_train, y_train)
    >>> save_model(model, 'models/regression_model.pkl')
    """
    # Create directory if it doesn't exist
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the model
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")

    return filepath


def load_model(filepath):
    """
    Load a model from a pickle file.

    Parameters:
    -----------
    filepath : str
        Path to the saved model file

    Returns:
    --------
    object : The loaded model

    Example:
    --------
    >>> model = load_model('models/regression_model.pkl')
    >>> predictions = model.predict(X_test)
    """
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def save_model_artifacts(model, scaler, features, base_path, model_name):
    """
    Save a model along with its scaler and feature list.

    This is a convenience function that saves all the artifacts needed
    for making predictions in one call.

    Parameters:
    -----------
    model : object
        The trained model
    scaler : object
        The fitted scaler (e.g., StandardScaler)
    features : list
        List of feature names
    base_path : str
        Directory where artifacts will be saved (e.g., 'models')
    model_name : str
        Prefix for the saved files (e.g., 'regression' or 'classification')

    Returns:
    --------
    dict : Paths to all saved artifacts

    Example:
    --------
    >>> save_model_artifacts(
    ...     model=my_model,
    ...     scaler=my_scaler,
    ...     features=X.columns.tolist(),
    ...     base_path='models',
    ...     model_name='regression'
    ... )
    """
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)

    paths = {
        'model': str(base / f"{model_name}_model.pkl"),
        'scaler': str(base / f"{model_name}_scaler.pkl"),
        'features': str(base / f"{model_name}_features.pkl")
    }

    joblib.dump(model, paths['model'])
    joblib.dump(scaler, paths['scaler'])
    joblib.dump(features, paths['features'])

    print(f"Saved {model_name} artifacts:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")

    return paths


def load_model_artifacts(base_path, model_name):
    """
    Load a model along with its scaler and feature list.

    Parameters:
    -----------
    base_path : str
        Directory where artifacts are saved
    model_name : str
        Prefix of the saved files (e.g., 'regression' or 'classification')

    Returns:
    --------
    dict : Dictionary containing 'model', 'scaler', and 'features'

    Example:
    --------
    >>> artifacts = load_model_artifacts('models', 'regression')
    >>> model = artifacts['model']
    >>> scaler = artifacts['scaler']
    >>> features = artifacts['features']
    """
    base = Path(base_path)

    artifacts = {
        'model': joblib.load(base / f"{model_name}_model.pkl"),
        'scaler': joblib.load(base / f"{model_name}_scaler.pkl"),
        'features': joblib.load(base / f"{model_name}_features.pkl")
    }

    print(f"Loaded {model_name} artifacts")

    return artifacts


def predict_with_scaling(model, scaler, input_data):
    """
    Make a prediction with automatic scaling.

    Parameters:
    -----------
    model : object
        The trained model
    scaler : object
        The fitted scaler
    input_data : DataFrame or array
        Input features to predict on

    Returns:
    --------
    array : Model predictions

    Example:
    --------
    >>> prediction = predict_with_scaling(model, scaler, new_data)
    """
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    return prediction


# Example usage when running this file directly
if __name__ == "__main__":
    print("Model Helper Utilities")
    print("=" * 40)
    print("\nAvailable functions:")
    print("  - save_model(model, filepath)")
    print("  - load_model(filepath)")
    print("  - save_model_artifacts(model, scaler, features, base_path, model_name)")
    print("  - load_model_artifacts(base_path, model_name)")
    print("  - predict_with_scaling(model, scaler, input_data)")
    print("\nSee docstrings for usage examples.")
