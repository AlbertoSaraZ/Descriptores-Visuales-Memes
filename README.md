# Descriptores-Visuales-Memes

*Computer.py son todos los descriptores visuales, subclases de la clase DescriptorComputer

descExtractor.py extrae las características y las almacena en un archivo .h5

main.py evalúa los descriptores/clasificadores según los hiperparámetros que se pongan, y entrega varias métricas de los resultados

hyperparam.py escanéa el espacio de hiperparámetros para cada par descriptor/clasificador según el espacio que uno le ponga, y dice cual es el mejor

resnetclassifier.py instancia resnet152 y deja que esta haga el proceso completo de clasificación

memeoracle.py entrena todos los clasificadores/descriptores y después ve en el conjunto de entrenamiento cuales imágenes no pudieron ser clasificadas por nada ninguna
