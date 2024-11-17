from concrete.ml.sklearn import (
    LogisticRegression, LinearRegression,
    DecisionTreeClassifier, RandomForestClassifier,
    NeuralNetClassifier, KNeighborsClassifier
)
from concrete.ml.deployment import FHEModelClient, FHEModelServer, FHEModelDev
import time
import os
import shutil

class ModelTrainer:
    def __init__(self, model_type: str = 'logistic_regression', n_bits: int = 7):
        self.n_bits = n_bits
        self.model = self.initialize_model(model_type)
        self.training_time = None
        self.compilation_time = None
        self.model_directory = None
        self.client = None
        self.server = None

    def initialize_model(self, model_type: str):
        if model_type == 'logistic_regression':
            return LogisticRegression(n_bits=self.n_bits)
        elif model_type == 'linear_regression':
            return LinearRegression(n_bits=self.n_bits)
        elif model_type == 'decision_tree':
            return DecisionTreeClassifier(n_bits=self.n_bits)
        elif model_type == 'random_forest':
            return RandomForestClassifier(n_bits=self.n_bits)
        elif model_type == 'mlp':
            return NeuralNetClassifier(
                module__n_layers=2,  # Number of hidden layers
                module__n_w_bits=self.n_bits,  # Weight quantization bits
                module__n_a_bits=self.n_bits,  # Activation quantization bits
                module__n_accum_bits=16,  # Accumulation bits
                max_epochs=50,
                batch_size=32,
                lr=0.01,
                #verbose=1
            )
        elif model_type == 'knn':
            return KNeighborsClassifier(n_bits=self.n_bits, n_neighbors=5)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

    def compile(self, representative_data, model_name='fhe_model'):
        start_time = time.time()
        self.model.compile(representative_data)
        self.compilation_time = time.time() - start_time

        # Save the compiled model
        self.model_directory = f'./models/{model_name}'
        # Create a backup if the directory exists
        if os.path.exists(self.model_directory):
            backup_directory = self.model_directory + '_backup'
            if os.path.exists(backup_directory):
                shutil.rmtree(backup_directory)
            shutil.copytree(self.model_directory, backup_directory)

        # Clear the path
        shutil.rmtree(self.model_directory, ignore_errors=True)

        # Save the model
        dev = FHEModelDev(path_dir=self.model_directory, model=self.model)
        dev.save()

    def load_fhe_model(self):
        if self.model_directory is None:
            raise ValueError("Model directory is not set. Compile the model first.")
        # Initialize FHEModelClient and FHEModelServer
        self.client = FHEModelClient(self.model_directory)
        self.server = FHEModelServer(self.model_directory)
        self.server.load()

    def predict(self, X_test):
        return self.model.predict(X_test)

    def fhe_predict(self, sample_text, data_processor):
        # Ensure client and server are initialized
        if self.client is None or self.server is None:
            self.load_fhe_model()

        # Transform and scale the sample text
        sample_vector = data_processor.vectorizer.transform([sample_text]).toarray()
        sample_vector_scaled = data_processor.scaler.transform(sample_vector)
        sample_vector_quantized = (sample_vector_scaled * data_processor.scale_factor).astype('float32')

        # Measure encryption time
        start_time = time.time()
        encrypted_input = self.client.quantize_encrypt_serialize(sample_vector_scaled)
        encryption_time = time.time() - start_time

        # Measure inference time
        start_time = time.time()
        serialized_evaluation_keys = self.client.get_serialized_evaluation_keys()
        encrypted_result = self.server.run(encrypted_input, serialized_evaluation_keys)
        inference_time = time.time() - start_time

        # Measure decryption time
        start_time = time.time()
        result = self.client.deserialize_decrypt_dequantize(encrypted_result)
        decryption_time = time.time() - start_time

        # Convert probabilities to label
        predicted_label = int(result.argmax())

        performance = {
            'encryption_time': encryption_time,
            'inference_time': inference_time,
            'decryption_time': decryption_time
        }

        return predicted_label, performance
