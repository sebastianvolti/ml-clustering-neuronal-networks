import logging
import numpy as np
import pandas as pd

from attributes import CONTINUOUS_ATTRS, DATASET_ATTRS
from sklearn import metrics, preprocessing


def replace_with_nan(dataset):
    values = [-1, -2, -3, -4, -5, -6, -7, -8, -9]
    for val in values:
        dataset = dataset.replace(to_replace=val, value=np.NaN)
    return dataset

def label(dataset, attrs):
    le = preprocessing.LabelEncoder()
    dataset[attrs] = dataset[attrs].apply(le.fit_transform)
    return dataset


def replace_most_common_val(dataset, attrs):
    for attr in attrs:
        most_common = dataset[attr].value_counts().index[0]
        dataset[attr] = dataset[attr].fillna(most_common)


def attributes_with_null(dataset):
    return [col_name for col_name, col in dataset.iteritems() if col.isna().sum() != 0]


def ordinal(dataset, attr):
    le = preprocessing.OrdinalEncoder(categories=[
                                        ["%s." % x for x in range(1, 29)],
                                        ['1. Less than H.S', '2. High School', '3. Some College',
                                         "4. Bachelor's", '5. Graduate Degree'],
                                        ['1. 17-20', '2. 21-24', '3. 25-29', '4. 30-34', '5. 35-39',
                                         '6. 40-44', '7. 45-49', '8. 50-54', '9. 55-59', '10. 60-64',
                                         '11. 65-69', '12. 70-74', '13. 75+']])
    dataset[attr] = le.fit(dataset[attr]).transform(dataset[attr])

    return dataset


def one_hot(dataset, attrs):
    dataset = pd.get_dummies(dataset, prefix=attrs, columns=attrs, drop_first=True)
    return dataset


def preprocess_dataset(dataset, ideology_groups, labelling=True):
    dataset = replace_with_nan(dataset)
    attrs = attributes_with_null(dataset)
    replace_most_common_val(dataset, attrs)
    if (ideology_groups > 0):
        dataset[['ideology']] = generalize_ideology(dataset[['ideology']])

    dataset = dataset.drop('case-id', axis=1)
    if labelling:
        dataset = label(dataset, ['race', 'religion', 'income', 'education', 'age-group', 'ideology'])
    return dataset


def normalize(dataset):
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset[['income', 'education', 'age-group','race', 'religion']] =  min_max_scaler.fit_transform(dataset[['income', 'education', 'age-group','race', 'religion']])
    return dataset


def generalize_ideology(dataset):
    liberal_values = [1,2,3]
    conservative_values = [5,6,7]
    moderate_value = [4]
    for val in liberal_values:
        dataset = dataset.replace(to_replace=liberal_values, value=1)
    for val in moderate_value:
        dataset = dataset.replace(to_replace=moderate_value, value=2)
    for val in conservative_values:
        dataset = dataset.replace(to_replace=conservative_values, value=3)
    print(dataset)
    return dataset


def generalize_income(dataset):
    for val in range(1, 29):
        if (val <= 10):
            dataset = dataset.replace(to_replace=val, value=1)
        elif (val <= 22):
            dataset = dataset.replace(to_replace=val, value=2)
        else:
            dataset = dataset.replace(to_replace=val, value=3)
    return dataset


def generalize_age(dataset):
    for val in range(1, 14):
        if (val <= 3):
            dataset = dataset.replace(to_replace=val, value=1)
        elif (val <= 9):
            dataset = dataset.replace(to_replace=val, value=2)
        else:
            dataset = dataset.replace(to_replace=val, value=3)
    return dataset



def classify_examples(classifier, evaluation_examples):
    obtained_classes = pd.DataFrame(classifier.predict(evaluation_examples), columns=['ideology'])

    return obtained_classes


# Auxiliares para particionar el dataset

def partition_sets(data, training_set_percentage, shuffle=True, no_class=False):
    ''' Mezcla los datos de entrenamiento y los separo en conjunto de entrenamiento y validación.
        no_class indica si hay instancias sin clasificación, y si es True, estas son insertadas
        dentro el conjunto de evaluación. El procentaje es calculado sobre el resto de instancias. '''
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True) # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    if no_class:
        no_class_set = data[data['ideology'].isnull()]
        data = data[~data['ideology'].isnull()]
    training_set_size = int(len(data) * training_set_percentage)
    training_set = data[:training_set_size].reset_index(drop=True)
    test_set = data[training_set_size:].reset_index(drop=True)
    pd.concat([no_class_set, test_set])
    return training_set, test_set


def get_classes(data):
    ''' Separa los datos de entrenamiento en dos listas distintas.
    Para cada ejemplo: [attr1, attr2, ..., attrN, class] en [attr1, attr2, ..., attrN] | [class] '''
    classes = data[['ideology']].copy(deep=True)
    examples = data.drop(columns='ideology')

    return examples, classes


# Auxiliares para métricas

def get_accuracy(obtained_classes, test_classes):
    return metrics.accuracy_score(test_classes, obtained_classes)

def get_precision(obtained_classes, test_classes, average='binary'):
    return metrics.precision_score(test_classes, obtained_classes, average=average)


def get_recall(obtained_classes, test_classes, average='binary'):
    return metrics.recall_score(test_classes, obtained_classes, average=average)


def get_f1_score(obtained_classes, test_classes, average='binary'):
    return metrics.f1_score(test_classes, obtained_classes, average=average)


def get_metrics(obtained_classes, evaluation_classes, average='binary'):
    acc = get_accuracy(obtained_classes, evaluation_classes)
    prec = get_precision(obtained_classes, evaluation_classes, average=average)
    rec = get_recall(obtained_classes, evaluation_classes, average=average)
    f1 = get_f1_score(obtained_classes, evaluation_classes, average=average)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }


def print_metrics(data):
    logging.info(f"> Accuracy: {data['accuracy']*100:.2f}%")
    logging.info(f"> Precision: {data['precision']*100:.2f}%")
    logging.info(f"> Recall: {data['recall']*100:.2f}%")
    logging.info(f"> F1: {data['f1']*100:.2f}%")


def preprocess_ideology(dataset_num):
    ido =[1.0,2.0,3.0,4.0,5.0,6.0,7.0]
    for val in ido:
        if val < 4.0 :
            dataset_num['ideology'] = dataset_num['ideology'].replace(to_replace=val, value=1.0)
        elif val == 4.0 :
            dataset_num['ideology'] = dataset_num['ideology'].replace(to_replace=val, value=2.0)
        else:
            dataset_num['ideology'] = dataset_num['ideology'].replace(to_replace=val, value=3.0)
            