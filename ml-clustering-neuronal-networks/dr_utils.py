import matplotlib.pyplot as plt
import pandas as pd
import prince
import utils

from attributes import DATASET_FILE, DATASET_FILE_NUM, DATASET_HEADERS
from matplotlib import colors as plt_colors
from sklearn import decomposition


def get_pca(dataset, n_components=2):
    pca = decomposition.PCA(n_components=n_components)
    principal_components = pca.fit_transform(dataset)
    principal_df = pd.DataFrame(data=principal_components)
    # final_df = pd.concat([principal_df, dataset[['ideology']]], axis=1)
    return principal_df


def get_mca(dataset, n_components=2):
    mca = prince.MCA(
     n_components=n_components,
     n_iter=3,
     copy=True,
     check_input=True,
     engine='auto',
     random_state=42
    )
    mca = mca.fit(dataset).transform(dataset)
    return mca


def get_famd(dataset, n_components=2):
    famd = prince.FAMD(
     n_components=n_components,
     n_iter=3,
     copy=True,
     check_input=True,
     engine='auto',
     random_state=42
    )
    famd = famd.fit(dataset).transform(dataset)
    return famd


def plot_mca(mca, dataset):
    ax = mca.plot_coordinates(
     X=dataset,
     ax=None,
     figsize=(6, 6),
     show_row_points=True,
     row_points_size=10,
     show_row_labels=False,
     show_column_points=True,
     column_points_size=30,
     show_column_labels=False,
     legend_n_cols=1
    )
    ax.get_figure().savefig('images/mca_coordinates.svg')


def plot_clusters(dataset, col_name_1, col_name_2):
    X = dataset[[col_name_1, col_name_2]]
    y = dataset['ideology']
    colors = ['r', 'r', 'r', 'g', 'b', 'b', 'b']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ideologies = ['1. Extremely Liberal', '2. Liberal', '3. Slightly Liberal' '4. Moderate',
                  '5. Slightly Conservative', '6. Conservative', '7. Extremely Conservative']

    for i, ideology in enumerate(ideologies):
        dt = dataset[dataset['ideology'] == ideology]
        ax.scatter(dt[col_name_1], dt[col_name_2], s=5, c=colors[i])


    plt.legend()
    plt.show()


def plot_clusters_3d(dataset, col_name_1, col_name_2, col_name_3):
    X = dataset[[col_name_1, col_name_2, col_name_3]]
    y = dataset['ideology']
    colors = ['r', 'r', 'r', 'g', 'b', 'b', 'b']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ideologies = ['1. Extremely Liberal', '2. Liberal', '3. Slightly Liberal' '4. Moderate',
                  '5. Slightly Conservative', '6. Conservative', '7. Extremely Conservative']

    for i, ideology in enumerate(ideologies):
        dt = dataset[dataset['ideology'] == ideology]
        ax.scatter(dt[col_name_1], dt[col_name_2], dt[col_name_3], s=5, c=colors[i])


    plt.legend()
    plt.show()


def prepare_data():
    # Lectura del csv común
    dataset = pd.read_csv(DATASET_FILE, delimiter=',', names=DATASET_HEADERS, header=None, engine='python')

    # Lectura del csv numérico
    dataset_num = pd.read_csv(DATASET_FILE_NUM, delimiter=',', names=DATASET_HEADERS, header=None, engine='python')
    
    # Preprocesamiento del conjunto de datos
    dataset = utils.preprocess_dataset(dataset, ideology_groups=0, labelling=False)
    dataset_num = utils.preprocess_dataset(dataset_num, ideology_groups=0, label=False)

    # Particionamiento del dataset
    TSET_PERCENTAGE = 0.8

    training_set, evaluation_set = utils.partition_sets(dataset, TSET_PERCENTAGE, no_class=True)
    training_examples, training_classes = utils.get_classes(training_set)
    evaluation_examples, evaluation_classes = utils.get_classes(evaluation_set)

    training_set_num, evaluation_set_num = utils.partition_sets(dataset_num, TSET_PERCENTAGE, no_class=True)
    training_examples_num, training_classes_num = utils.get_classes(training_set_num)
    evaluation_examples_num, evaluation_classes_num = utils.get_classes(evaluation_set_num)

    return training_examples, training_classes, training_examples_num, training_classes_num
