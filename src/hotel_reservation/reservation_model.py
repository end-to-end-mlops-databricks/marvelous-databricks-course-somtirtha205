from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score,confusion_matrix

class ReservationModel:
    def __init__(self, preprocessor, config):
        self.config = config
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=config['parameters']['n_estimators'],
                max_depth=config['parameters']['max_depth'],
                random_state=42
            ))
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        return f1, cm

    def get_feature_importance(self):
        feature_importance = self.model.named_steps['classifier'].feature_importances_
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        return feature_importance, feature_names