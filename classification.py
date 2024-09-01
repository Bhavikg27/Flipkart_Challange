from sklearn import svm
from sklearn.model_selection import train_test_split

def train_model(X_train, y_train):
    """Trains a Support Vector Machine (SVM) model."""
    model = svm.SVC(kernel='linear')  # Initialize an SVM with a linear kernel
    model.fit(X_train, y_train)  # Train the model with training data
    return model

def predict(model, X_test):
    """Uses the trained model to make predictions on test data."""
    predictions = model.predict(X_test)  # Predict using the trained model
    return predictions
