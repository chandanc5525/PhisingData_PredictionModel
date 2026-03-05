import logging
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline


class DataTransformation:

    def __init__(self):
        pass

    def get_preprocessor(self, X):

        numerical_cols = X.select_dtypes(exclude="object").columns
        categorical_cols = X.select_dtypes(include="object").columns

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

        return preprocessor

    def transform(self, X_train, X_test, y_train):

        logging.info("Starting Data Transformation")

        preprocessor = self.get_preprocessor(X_train)

        # Step 1: Preprocess the data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Step 2: Apply SMOTE to training data only
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_processed, y_train
        )

        # Step 3: Apply PCA after SMOTE
        pca = PCA(n_components=0.95,k_neighbors=3)
        X_train_transformed = pca.fit_transform(X_train_resampled)
        X_test_transformed = pca.transform(X_test_processed)

        logging.info("Data Transformation Completed")

        return X_train_transformed, X_test_transformed, y_train_resampled