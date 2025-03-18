import tensorflow as tf
import os
import matplotlib.pyplot as plt
import  numpy as np

BASE_DIR = r'.\data'  # Use raw string (r'...') to avoid escape sequence issues

train_dir = os.path.join(BASE_DIR, 'dataset')

begin_training = False


CLASS_NAMES = []
for root, dirs, files in os.walk(train_dir):
    # Append only the first level of subdirectories
    CLASS_NAMES.extend(dirs)
    break  # Break after processing the top level

print("Class Names:", CLASS_NAMES)
NUM_CLASSES = len(CLASS_NAMES)
print(f"Number of directories: {NUM_CLASSES}")


# Get the number of classes (Counting the directories)
#NUM_CLASSES = len([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])


# Check if the directory exists
if os.path.exists(train_dir):
    print(f"Directory exists: {train_dir}")
    begin_training = True
else:
    print(f"Directory does not exist: {train_dir}")
    begin_training = False

def plot_loss_acc(history):
    '''Plots the training and validation loss and accuracy from a history object'''
    acc = history.history['accuracy']
    loss = history.history['loss']

    epochs = range(len(acc))

    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    ax[0].plot(epochs, acc, 'bo', label='Training accuracy')
    ax[0].set_title('Training accuracy')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('accuracy')
    ax[0].legend()

    ax[1].plot(epochs, loss, 'bo', label='Training Loss')
    ax[1].set_title('Training loss')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('loss')
    ax[1].legend()

    plt.show()

# Instantiate the training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(110, 110),
    batch_size=3,
    label_mode='categorical'
    )

# Create the augmentation model.
FILL_MODE = 'nearest'
data_augmentation = tf.keras.Sequential([
    tf.keras.Input(shape=(110, 110, 3)),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2, fill_mode=FILL_MODE),
    tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode=FILL_MODE),
    tf.keras.layers.RandomZoom(0.2, fill_mode=FILL_MODE),
    tf.keras.layers.RandomContrast(0.2)  # Add RandomContrast
])


# Load the pre-trained MobileNetV2 model without the top classification layer
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(110, 110, 3),  # Image size suitable for MobileNetV2
    include_top=False,          # Exclude the top classification layers
    weights='imagenet'          # Use ImageNet pre-trained weights
)

# Freeze the layers of the base model (don't train them)
base_model.trainable = False

# Create the custom model with face recognition layers on top
model = tf.keras.models.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),  # Pooling layer to reduce dimensions
    tf.keras.layers.Dense(1024, activation='relu'),  # Fully connected layer
    tf.keras.layers.Dropout(0.5),  # Dropout to prevent overfitting
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer: NUM_CLASSES is the number of faces
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Cross-entropy loss for classification
              metrics=['accuracy'])

# Show the model summary
model.summary()

# Train the model
history = model.fit(train_dataset, epochs=15)
plot_loss_acc(history)

# Make a prediction
image_path = r'F:\AIDeployments\FaceID-Replica\data\test3\forward.jpg'

image = tf.keras.utils.load_img(image_path, target_size=(110, 110))
image = tf.keras.utils.img_to_array(image) # convert to np_array
image = np.expand_dims(image, axis=0) # add the batch dimension

# Make the prediction
predictions = model.predict(image)
print("Raw predictions:", predictions)
predicted_class_index = np.argmax(predictions)  # Get the index of the highest probability
predicted_class = CLASS_NAMES[predicted_class_index]
print("Predicted class:", predicted_class)
