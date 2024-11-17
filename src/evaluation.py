from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import os
import time
import random
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm

class Evaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred)
        self.metrics['recall'] = recall_score(y_true, y_pred)
        self.metrics['f1_score'] = f1_score(y_true, y_pred)
        self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        return self.metrics

    def measure_prediction_time(self, model, X_test):
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        return y_pred, prediction_time

    def measure_fhe_prediction_performance(self, model_trainer, X_test_text, y_test, data_processor, sample_ratio=0.1):
        """
        Measure the performance of the FHE model on a subset of the test data.
        This method is reimplementation of the `fhe_predict` method in the `ModelTrainer` class.
        It includes sample subsetting for input, time measurements for encryption, inference, and decryption.
        """
        # Sample a subset of indices for faster evaluation
        random.seed(42)
        subset_indices = random.sample(range(len(X_test_text)), int(len(X_test_text) * sample_ratio))

        # Use positional indexing for both X_test_text and y_test
        X_test_text_subset = [X_test_text.iloc[i] for i in subset_indices]
        y_test_subset = [y_test.iloc[i] for i in subset_indices]

        # Batch process: transform and scale inputs for encryption
        X_test_vectors = data_processor.vectorizer.transform(X_test_text_subset).toarray()
        X_test_scaled = data_processor.scaler.transform(X_test_vectors)
        X_test_quantized = (X_test_scaled * data_processor.scale_factor).astype('float32')
        
        # Reshape all samples at once to (num_samples, 1, num_features)
        X_test_quantized = X_test_quantized.reshape(-1, 1, X_test_quantized.shape[1])
        
        # Measure batch encryption time
        start_time = time.time()
        encrypted_inputs = [
            model_trainer.client.quantize_encrypt_serialize(sample)
            for sample in X_test_quantized
        ]
        encryption_time = time.time() - start_time


        # Measure batch inference time with parallelism
        start_time = time.time()
        serialized_evaluation_keys = model_trainer.client.get_serialized_evaluation_keys()

        # Define the inference function
        def infer_encrypted_input(encrypted_input):
            start_time = time.time()
            result = model_trainer.server.run(encrypted_input, serialized_evaluation_keys)
            inference_time = time.time() - start_time
            return result, inference_time

        # Use ThreadPoolExecutor for parallel execution
        max_workers = min(8, (os.cpu_count() or 1) * 2)
        inference_times = []
        encrypted_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(infer_encrypted_input, encrypted_input)
                for encrypted_input in encrypted_inputs
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Inference Progress"):
                try:
                    result, inference_time = future.result()
                    encrypted_results.append(result)
                    inference_times.append(inference_time)
                except Exception as e:
                    print(f"Error during inference: {e}")

        # Measure batch decryption time
        start_time = time.time()
        decrypted_results = [
            model_trainer.client.deserialize_decrypt_dequantize(encrypted_result)
            for encrypted_result in encrypted_results
        ]
        decryption_time = time.time() - start_time

        # Convert probabilities to labels
        y_pred = [int(result.argmax()) for result in decrypted_results]

        # Store performance metrics
        self.metrics['fhe_encryption_time'] = encryption_time / len(X_test_quantized)
        self.metrics['fhe_inference_time'] = sum(inference_times) / len(inference_times)
        self.metrics['fhe_decryption_time'] = decryption_time / len(X_test_quantized)

        return y_pred, y_test_subset
