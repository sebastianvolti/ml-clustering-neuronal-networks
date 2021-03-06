import dr_utils
import pandas as pd
import numpy as np
import argparse
import knn
import logging
import random
import utils
import dataset_analysis
import means
import means_plot
from attributes import CONTINUOUS_ATTRS, DATASET_ATTRS


################## Config básica ##################

DEBUG = True

# Lee argumentos
ap = argparse.ArgumentParser(
    description='Tarea de AprendAut')
ap.add_argument('-s', '--seed', default=3, help='Indica la semilla a utilizar para la librería random')
ap.add_argument('-d', '--debug_level', default=2, help='0 si no se quiere imprimir nada, 1 para mensajes de info, 2 para mensajes de debug')
ap.add_argument('-v', '--view', default='2d', help='Toma valores 2d o 3d')

args = vars(ap.parse_args())
seed = int(args['seed'])
debug_level = int(args['debug_level'])
view = args['view']

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
    np.random.seed(seed) # Seteo la seed para numpy (que es lo que usa scikit)

    training_examples, training_classes, training_examples_num, training_classes_num = dr_utils.prepare_data()

    if view == '2d':
        # PCA
        # Directo PCA con datos numéricos
        training_set_pca = dr_utils.get_pca(training_examples_num, n_components=2)
        dr_utils.plot_clusters(pd.concat([training_set_pca, training_classes], axis=1), 0, 1)

        # MCA
        # MCA directo con datos string
        training_set_mca = dr_utils.get_mca(training_examples, n_components=2)
        # dr_utils.plot_clusters(pd.concat([training_set_mca, training_classes], axis=1), 0, 1)
        dr_utils.plot_clusters(pd.concat([training_set_mca, training_classes], axis=1), 0, 1)

        # MCA con representación One Hot
        training_examples_oe = utils.one_hot(training_examples, DATASET_ATTRS)
        training_set_mca = dr_utils.get_mca(training_examples_oe, n_components=2)
        dr_utils.plot_clusters(pd.concat([training_set_mca, training_classes], axis=1), 0, 1)

        # FAMD
        # FAMD directo con datos string
        training_set_famd = dr_utils.get_famd(training_examples, n_components=2)
        dr_utils.plot_clusters(pd.concat([training_set_famd, training_classes], axis=1), 0, 1)

        # FAMD con datos ordinales pasados a numéricos
        training_examples_ord = utils.ordinal(training_examples, CONTINUOUS_ATTRS)
        training_set_famd = dr_utils.get_famd(training_examples_ord, n_components=2)
        dr_utils.plot_clusters(pd.concat([training_set_famd, training_classes], axis=1), 0, 1)
    else:
        # PCA
        # Directo PCA con datos numéricos
        training_set_pca = dr_utils.get_pca(training_examples_num, n_components=3)
        dr_utils.plot_clusters_3d(pd.concat([training_set_pca, training_classes], axis=1), 0, 1, 2)

        # MCA
        # MCA directo con datos string
        training_set_mca = dr_utils.get_mca(training_examples, n_components=3)
        # dr_utils.plot_clusters(pd.concat([training_set_mca, training_classes], axis=1), 0, 1)
        dr_utils.plot_clusters_3d(pd.concat([training_set_mca, training_classes], axis=1), 0, 1, 2)

        # MCA con representación One Hot
        training_examples_oe = utils.one_hot(training_examples, DATASET_ATTRS)
        training_set_mca = dr_utils.get_mca(training_examples_oe, n_components=3)
        dr_utils.plot_clusters_3d(pd.concat([training_set_mca, training_classes], axis=1), 0, 1, 2)

        # FAMD
        # FAMD directo con datos string
        training_set_famd = dr_utils.get_famd(training_examples, n_components=3)
        dr_utils.plot_clusters_3d(pd.concat([training_set_famd, training_classes], axis=1), 0, 1, 2)

        # FAMD con datos ordinales pasados a numéricos
        training_examples_ord = utils.ordinal(training_examples, CONTINUOUS_ATTRS)
        training_set_famd = dr_utils.get_famd(training_examples_ord, n_components=3)
        dr_utils.plot_clusters_3d(pd.concat([training_set_famd, training_classes], axis=1), 0, 1, 2)


if __name__ == "__main__":
   main()
