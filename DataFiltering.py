import re
from collections import defaultdict
import math
import random

# Cargar dataset manualmente sin librerías externas
def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = [line.strip().split('\t', 1) for line in lines]
    return [(label, message) for label, message in data]

dataset = load_dataset("entrenamiento.txt")

# Normalización de texto (eliminar caracteres especiales y unificar mayúsculas/minúsculas)
def clean_text(text):
    text = text.lower().strip()  # Convertir a minúsculas y eliminar espacios
    text = re.sub(r'[^a-z0-9 ]', '', text)  # Eliminar caracteres especiales
    return text

dataset = [(label, clean_text(message)) for label, message in dataset]

# Separar dataset en 80% training y 20% testing
random.seed(42)
random.shuffle(dataset)
split_idx = int(0.8 * len(dataset))
training_data, testing_data = dataset[:split_idx], dataset[split_idx:]

# Guardar datasets manualmente
def save_dataset(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for label, message in data:
            f.write(f"{label}\t{message}\n")

save_dataset("lol_training.txt", training_data)
save_dataset("lol_testing.txt", testing_data)

print("Training size:", len(training_data))
print("Testing size:", len(testing_data))

# Implementación de Naive Bayes con Laplace Smoothing
class NaiveBayes:
    def __init__(self):
        self.word_probs = {}
        self.class_probs = {}
        self.vocab = set()
    
    def fit(self, data):
        class_counts = defaultdict(lambda: 0)
        word_counts = defaultdict(lambda: defaultdict(lambda: 0))
        total_docs = len(data)
        
        for label, message in data:
            class_counts[label] += 1
            words = message.split()
            self.vocab.update(words)
            for word in words:
                word_counts[label][word] += 1
        
        self.class_probs = {cls: count / total_docs for cls, count in class_counts.items()}
        vocab_size = len(self.vocab)
        
        for cls in class_counts:
            total_words = sum(word_counts[cls].values())
            self.word_probs[cls] = {word: (word_counts[cls][word] + 1) / (total_words + vocab_size)
                                    for word in self.vocab}
    
    def predict(self, messages):
        predictions = []
        for message in messages:
            words = message.split()
            class_scores = {cls: math.log(prob) for cls, prob in self.class_probs.items()}
            
            for cls in self.class_probs:
                for word in words:
                    if word in self.word_probs[cls]:
                        class_scores[cls] += math.log(self.word_probs[cls][word])
            
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions

# Entrenar modelo
nb_model = NaiveBayes()
nb_model.fit(training_data)

# Evaluación del modelo
test_messages = [message for _, message in testing_data]
test_labels = [label for label, _ in testing_data]
test_predictions = nb_model.predict(test_messages)
accuracy = sum(1 for true, pred in zip(test_labels, test_predictions) if true == pred) / len(test_labels)
print(f"Naive Bayes Accuracy: {accuracy:.4f}")
