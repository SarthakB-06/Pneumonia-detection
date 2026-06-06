import os
import tensorflow as tf
from pathlib import Path
import numpy as np

# Import our custom modules
from src import config
from src.data_loader import create_data_generators
from src.model import build_model

def get_dynamic_class_weights(train_dir):
    """Dynamically calculates weights based on current folder counts."""
    num_normal = len(list(Path(train_dir / "NORMAL").glob('*')))
    num_pneumonia = len(list(Path(train_dir / "PNEUMONIA").glob('*')))
    total = num_normal + num_pneumonia
    
    weight_0 = total / (2 * num_normal)
    weight_1 = total / (2 * num_pneumonia)
    
    print(f"Computed Class Weights -> Normal: {weight_0:.2f}, Pneumonia: {weight_1:.2f}")
    return {0: weight_0, 1: weight_1}

def main():
    print("--- Starting Production Training Pipeline ---")
    
    # 1. Load Data
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir=config.TRAIN_DATA,
        val_dir=config.VAL_DATA,
        test_dir=config.TEST_DATA,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )    
    # 2. Compute Class Weights dynamically
    class_weights = get_dynamic_class_weights(config.TRAIN_DATA)
    
    # 3. Build Model
    model = build_model()
    
    # 4. Set up Production Callbacks
    # Create a directory to save the trained model
    os.makedirs("saved_models", exist_ok=True)
    
    callbacks = [
        # Saves the model ONLY when validation recall improves
        tf.keras.callbacks.ModelCheckpoint(
            filepath="saved_models/best_pneumonia_model.keras",
            save_best_only=True,
            monitor="val_recall", 
            mode="max",
            verbose=1
        ),
        # Stops training if the model doesn't improve for 5 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 5. Execute Training
    print("\nBeginning Model Training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20, # We can set this high because EarlyStopping will cut it off if needed
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # 6. Final Evaluation on unseen Test Data
    print("\n--- Training Complete. Evaluating on Test Set ---")
    test_results = model.evaluate(test_generator)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")

if __name__ == "__main__":
    main()