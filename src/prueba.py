from clasificador import MosquitoClassifier

classifier = MosquitoClassifier(
    "../models/modelo_final_mosquitos.keras"
)

resultado = classifier.predict(
    "../data/raw/Mosquito Dataset Augmented/Aedes/nombredeimagen.jpg"
)

print(resultado)