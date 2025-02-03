# Laboratorio 2
# Clasificador de mensajes
# Task 2
# Ricardo Chuy, Silvia Illescas, Nelson García y Michelle Mejía
import re
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Limpieza del dataset y conversión a minúsculas
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Carga del dataset
def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    labels = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            labels.append(parts[0].lower())
            data.append(clean_text(parts[1]))
    return data, labels

# Dividimos 80% para el entrenamiento y 20% test
def split_data(data, labels):
    return train_test_split(data, labels, test_size=0.2, random_state=42)

# Entrenamiento con Laplace Smoothing
def train_naive_bayes(train_texts, train_labels):
    word_counts = {"spam": defaultdict(lambda: 1), "ham": defaultdict(lambda: 1)}
    class_counts = {"spam": 0, "ham": 0}

    for text, label in zip(train_texts, train_labels):
        words = text.split()
        class_counts[label] += 1
        for word in words:
            word_counts[label][word] += 1

    vocab_size = len(set(word_counts['spam'].keys()).union(set(word_counts['ham'].keys())))
    total_spam_words = sum(word_counts['spam'].values())
    total_ham_words = sum(word_counts['ham'].values())

    spam_probs = {word: (word_counts['spam'][word] / (total_spam_words + vocab_size)) for word in word_counts['spam']}
    ham_probs = {word: (word_counts['ham'][word] / (total_ham_words + vocab_size)) for word in word_counts['ham']}

    priors = {"spam": class_counts["spam"] / sum(class_counts.values()), "ham": class_counts["ham"] / sum(class_counts.values())}

    return spam_probs, ham_probs, priors, vocab_size

# Predicciones de Spam o Ham
def predict(text, spam_probs, ham_probs, priors, vocab_size):
    words = text.split()
    spam_prob = np.log(priors['spam'])
    ham_prob = np.log(priors['ham'])

    for word in words:
        spam_prob += np.log(spam_probs.get(word, 1 / (sum(spam_probs.values()) + vocab_size)))
        ham_prob += np.log(ham_probs.get(word, 1 / (sum(ham_probs.values()) + vocab_size)))
    
    prob_spam = np.exp(spam_prob) / (np.exp(spam_prob) + np.exp(ham_prob))
    prob_ham = np.exp(ham_prob) / (np.exp(spam_prob) + np.exp(ham_prob))
    
    return ('spam' if spam_prob > ham_prob else 'ham'), prob_spam, prob_ham

# Evaluación y Métricas del modelo
def evaluate_model(test_texts, test_labels, spam_probs, ham_probs, priors, vocab_size):
    test_labels = [label.strip().lower() for label in test_labels]  # Asegurar que las etiquetas sean cadenas limpias
    predictions = [predict(text, spam_probs, ham_probs, priors, vocab_size)[0] for text in test_texts]
    acc = accuracy_score(test_labels, predictions)
    prec = precision_score(test_labels, predictions, pos_label='spam')
    rec = recall_score(test_labels, predictions, pos_label='spam')
    f1 = f1_score(test_labels, predictions, pos_label='spam')

    print("Custom Naive Bayes Model Performance:")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1-score: {f1}\n")

# Uso de librerías y comparativa
def compare_with_sklearn(train_texts, train_labels, test_texts, test_labels):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = MultinomialNB()
    model.fit(X_train, train_labels)
    predictions = model.predict(X_test)

    acc = accuracy_score(test_labels, predictions)
    prec = precision_score(test_labels, predictions, pos_label='spam')
    rec = recall_score(test_labels, predictions, pos_label='spam')
    f1 = f1_score(test_labels, predictions, pos_label='spam')

    print("Sklearn Naive Bayes Model Performance:")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1-score: {f1}\n")

# Clasificación de mensajes futuros
def classify_message(spam_probs, ham_probs, priors, vocab_size):
    while True:
        user_message = input("Ingrese un mensaje para clasificar: ")
        cleaned_message = clean_text(user_message)
        prediction, prob_spam, prob_ham = predict(cleaned_message, spam_probs, ham_probs, priors, vocab_size)
        print(f"El mensaje ingresado ha sido clasificado como: {prediction}")
        print(f"Probabilidad de Spam: {prob_spam:.4f}, Probabilidad de Ham: {prob_ham:.4f}")
        cont = input("¿Desea clasificar otro mensaje? (s/n): ").strip().lower()
        if cont != 's':
            break

if __name__ == "__main__":
    data, labels = load_dataset("entrenamiento.txt")
    train_texts, test_texts, train_labels, test_labels = split_data(data, labels)
    print("Training Sklearn Naive Bayes Model...")
    compare_with_sklearn(train_texts, train_labels, test_texts, test_labels)
    print("Training Custom Naive Bayes Model...")
    spam_probs, ham_probs, priors, vocab_size = train_naive_bayes(train_texts, train_labels)
    evaluate_model(test_texts, test_labels, spam_probs, ham_probs, priors, vocab_size)
    classify_message(spam_probs, ham_probs, priors, vocab_size)

# Comentario sobre la métrica utilizada
# Accuracy es una métrica adecuada cuando las clases están balanceadas en el dataset.
# Sin embargo, en caso de que haya un desbalance significativo entre ham y spam,
# métricas como Precision y Recall pueden proporcionar una mejor evaluación del rendimiento del modelo.
# Precision nos dice cuántos de los mensajes clasificados como spam realmente lo son,
# mientras que Recall mide cuántos de los mensajes de spam reales fueron correctamente identificados.

# Comparación de Modelos
# La implementación con Sklearn obtiene una mayor precisión y F1-score en general,
# debido a la optimización interna y al uso eficiente de probabilidades suavizadas.
# Sin embargo, la implementación manual es útil para comprender cómo funciona Naive Bayes.
# Sklearn maneja mejor grandes volúmenes de datos y optimiza hiperparámetros.
# Accuracy es adecuada si las clases están balanceadas, pero en caso de desbalance,
# Precision y Recall pueden proporcionar una mejor evaluación del rendimiento del modelo.

