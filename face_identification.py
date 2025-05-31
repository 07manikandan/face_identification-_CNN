import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import pyttsx3
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# Load your CSV
df = pd.read_csv("data.csv")  # photo, name, age, gender

# Encode name as label
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['name'])

# Load and preprocess images
image_folder = "augmented_faces"
images = []
for img_name in df['photo']:
    img_path = os.path.join(image_folder, img_name)
    img = load_img(img_path, target_size=(128, 128))
    img = img_to_array(img) / 255.0
    images.append(img)

X = np.array(images)
print(X)
y = to_categorical(df['label_enc'])
print(y)
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model and encoder
model.save("face_id_model.h5")
np.save("label_encoder_classes.npy", le.classes_)

# //////////////////////////////////////////////////////////////

# Constants
IMG_SIZE = (100, 100)
IMAGE_FOLDER = "augmented_faces"
CSV_PATH = "data.csv"
MODEL_PATH = "cnn_feature_model.h5"
THRESHOLD = 0.90

# Text-to-Speech
engine = pyttsx3.init()
def speak(text):
    print("🔊", text)
    engine.say(text)
    engine.runAndWait()

# Load and preprocess image
def extract_feature(img_path, model):
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return model.predict(img)[0]

# Get model for extracting features
def get_feature_model(full_model):
    return Model(inputs=full_model.input, outputs=full_model.layers[-2].output)

# Load dataset and extract features
def load_dataset_features(df, feature_model):
    features = []
    for _, row in df.iterrows():
        img_path = os.path.join(IMAGE_FOLDER, row['photo'])
        if os.path.exists(img_path):
            feat = extract_feature(img_path, feature_model)
        else:
            feat = np.zeros(128)
        features.append(feat)
    df['features'] = features
    return df

# Find best match
def find_match(input_img_path, df, feature_model):
    input_feat = extract_feature(input_img_path, feature_model)
    dataset_feats = np.stack(df['features'].to_numpy())
    sims = cosine_similarity([input_feat], dataset_feats)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] >= THRESHOLD:
        return df.iloc[best_idx], sims[best_idx]
    return None, None

# Main
def main():
    print("🔍 Loading model and data...")
    model = load_model(MODEL_PATH)
    df = pd.read_csv(CSV_PATH)
    feature_model = get_feature_model(model)
    df = load_dataset_features(df, feature_model)

    input_img_path = input("📤 Enter input image path: ").strip()
    if not os.path.exists(input_img_path):
        print("❌ Image not found.")
        return

    match, score = find_match(input_img_path, df, feature_model)
    if match is not None:
        print("\n✅ Match Found!")
        print("Name   :", match['name'])
        print("Age    :", match['age'])
        print("Gender :", match['gender'])
        print(f"Similarity: {score:.2f}")
        speak(f"Match found. Name: {match['name']}, Age: {match['age']}, Gender: {match['gender']}")
    else:
        print("\n❌ No matching face found.")
        speak("No matching face found.")

if __name__ == "__main__":
    main()
