import pandas as pd
import numpy as np
import argparse
import copy
import knn
import logging
import statistics
import random
import utils
import dataset_analysis
import means
import means_plot

from attributes import DATASET_ATTRS, DATASET_FILE, DATASET_FILE_NUM, DATASET_HEADERS, DATASET_CLASS
from sklearn import neighbors


################## Config básica ##################

DEBUG = True

# Lee argumentos
ap = argparse.ArgumentParser(
    description='Tarea de AprendAut')
ap.add_argument('-s', '--seed', default=3, help='Indica la semilla a utilizar para la librería random')
ap.add_argument('-d', '--debug_level', default=2, help='0 si no se quiere imprimir nada, 1 para mensajes de info, 2 para mensajes de debug')
ap.add_argument('-p', '--part', help='Indica con valores a, b o c, qué parte de la tarea se quiere ejecutar')
ap.add_argument('-g', '--graficos', default=0, help='1 si se quieren mostrar los graficos, 0 si no')
ap.add_argument('-r', '--runs', default=500, help='Corridas de k-means buscando clusters optimos')
ap.add_argument('-l', '--limit', default=300, help='Limite de iteraciones por cada corrida de k-means')
ap.add_argument('-n', '--normalize', default=0, help='1 si se quiere trabajar con atributos normalizados, 0 si no')
ap.add_argument('-i', '--ideology_groups', default=0, help='1 si se quiere trabajar con ideologias agrupadas, 0 si no')

args = vars(ap.parse_args())
seed = int(args['seed'])
debug_level = int(args['debug_level'])
part = args['part']
plot_grafics = int(args['graficos'])
runs_k_means = int(args['runs'])
iters_k_means = int(args['limit'])
normalize = int(args['normalize'])
ideology_groups = int(args['ideology_groups'])


if debug_level == 0:
    logging_level = logging.WARNING
elif debug_level == 1:
    logging_level = logging.INFO
elif debug_level == 2:
    logging_level = logging.DEBUG

logging.basicConfig(level=logging_level, format='%(message)s')


################## Comienzo del main ##################


def main():

    random.seed(seed)
    np.random.seed(seed)

    # Lectura del csv numérico
    dataset_num = pd.read_csv(DATASET_FILE_NUM, delimiter=',', names=DATASET_HEADERS, header=None, engine='python')
    
    # Preprocesamiento del conjunto de datos
    dataset = utils.preprocess_dataset(dataset_num, ideology_groups)

    if part == 'a':
        # Analisis del conjunto de datos normal
        analysis = dataset_analysis.get_analysis(dataset, DATASET_ATTRS, DATASET_CLASS)
        print(analysis)

        # Analisis del conjunto de datos numérico
        analysis_num = dataset_analysis.get_analysis(dataset_num, DATASET_ATTRS, DATASET_CLASS)
        print(analysis_num)

        if (plot_grafics > 0):
            dataset_analysis.get_plots(dataset_num)
            pc_dataset = dataset_analysis.get_pca(dataset_num)
            dataset_analysis.plot_clusters(pc_dataset, 'pc1', 'pc2')

    elif part == 'c': 
        if (normalize > 0):
            dataset = utils.normalize(dataset)
        if (plot_grafics > 0):  
            means_plot.pruebita(dataset)
        #k-means
        means.means(dataset, runs_k_means, iters_k_means, plot_grafics)
   
    elif part == 'd':
        
        if (normalize > 0):
            dataset = utils.normalize(dataset)

        # Cross validation
        cv_examples, cv_classes = utils.get_classes(dataset)
        ks = [x for x in range(1, 44, 2)]
        best_k = ks[0]
        best_score = 0

        for k in ks:
            logging.info(f"Evaluando cross validation de KNN con k={k}")
            scores = knn.cross_val(neighbors.KNeighborsClassifier(n_neighbors=k), cv_examples, cv_classes.values.ravel(), scoring='accuracy', cv_folds=5)
            mean_score = statistics.mean(scores)
            logging.info(f"El accuracy promedio obtenido es {mean_score*100:.2f}%")
            if mean_score > best_score:
                best_k = k
                best_score = mean_score
            logging.info(f"")

        logging.info(f"El mejor k es {best_k}")
        logging.info(f"")

        # Particionamiento del dataset
        TSET_PERCENTAGE = 0.9
        training_set, evaluation_set = utils.partition_sets(dataset, TSET_PERCENTAGE, no_class=True)
        training_examples, training_classes = utils.get_classes(training_set)
        evaluation_examples, evaluation_classes = utils.get_classes(evaluation_set)

        # KNN
        classifier_knn = knn.scikit_knn(training_examples, training_classes, k=best_k)
        obtained_classes_knn = utils.classify_examples(classifier_knn, evaluation_examples)

        # Métricas
        logging.info(f"Resultados KNN Scikit")
        logging.info(f"Resultados micro")
        metrics = utils.get_metrics(obtained_classes_knn, evaluation_classes, average='micro')
        utils.print_metrics(metrics)
        logging.info(f"")

        logging.info(f"Resultados macro")
        metrics = utils.get_metrics(obtained_classes_knn, evaluation_classes, average='macro')
        utils.print_metrics(metrics)

    elif part == 'e':
        
        estratificacion=dataset['ideology'].value_counts()
        attrs=attrs = copy.deepcopy(DATASET_ATTRS)
        print(estratificacion)
        dataset_analysis.mean_var_std(dataset)
        dataset_analysis.histogram_vs_ideology(dataset,attrs)
        dataset_analysis.kdeplot_vs_ideology(dataset,attrs)
        dataset_analysis.Box_plot(dataset,attrs)


if __name__ == "__main__":
   main()
