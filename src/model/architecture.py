import tensorflow as tf
from src import config

def build_model():
    """
    Builds a Transfer Learning model using ResNet50V2.
    """
    print("Loading pre-trained ResNet50V2 base...")
    
    # 1. Load the pre-trained base model
    # include_top=False chops off the original 1000-class ImageNet classifier
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS),
        include_top=False,
        weights='imagenet'
    )
    
    # 2. Freeze the base model 
    # This prevents the pre-trained weights from being destroyed during initial training
    base_model.trainable = False
    
    # 3. Build our custom classification head
    print("Assembling custom classification layers...")
    model = tf.keras.Sequential([
        base_model,
        
        # Condenses the complex feature maps into a single 1D vector
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Randomly turns off 20% of neurons to prevent memorization (overfitting)
        tf.keras.layers.Dropout(0.2),
        
        # The final decision layer: output is between 0 and 1 (Normal vs Pneumonia)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 4. Compile the model
    # In medical AI, accuracy is not enough. We MUST track Recall (False Negatives).
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model