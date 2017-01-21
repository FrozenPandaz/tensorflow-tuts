import tempfile
import urllib.request

print('Requesting Training Data')
TRAIN_FILE, headers = urllib.request.urlretrieve(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
)
print('Requesting Test Data')
TEST_FILE, headers = urllib.request.urlretrieve(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
)

train_file = open(TRAIN_FILE)
test_file = open(TEST_FILE)

import pandas as pd

COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income_bracket'
]

df_train = pd.read_csv(
    train_file,
    names = COLUMNS,
    skipinitialspace=True
)

df_test = pd.read_csv(
    test_file,
    names=COLUMNS,
    skipinitialspace=True,
    skiprows = 1
)

# Cast to 0 or 1
LABEL_COLUMN = 'label'
df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)

CATEGORICAL_COLUMNS = [
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'native_country'
]

CONTINUOUS_COLUMNS = [
    'age',
    'education_num',
    'capital_gain',
    'capital_loss',
    'hours_per_week'
]

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values)
    for k in CONTINUOUS_COLUMNS}

    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1]
    )
    for k in CATEGORICAL_COLUMNS}

    feature_cols = dict(continuous_cols.items())
    feature_cols.update(dict(categorical_cols.items()))

    label = tf.constant(df[LABEL_COLUMN].values)

    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def test_input_fn():
    return input_fn(df_test)

def hash_layer(name, size):
    return tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name=name,
        hash_bucket_size=size
    )

def cont_layer(name):
    return tf.contrib.layers.real_valued_column(name)

# Categorical Columns
gender = tf.contrib.layers.sparse_column_with_keys(
    column_name='gender', keys=['Female', 'Male']
)
education = hash_layer('education', 1000)
relationship = hash_layer('relationship', 100)
workclass = hash_layer('workclass', 100)
occupation = hash_layer('occupation', 1000)
native_country = hash_layer('native_country', 1000)
marital_status = hash_layer('marital_status', 10)
race = hash_layer('race', 1000)

# Continuous Columns
age = cont_layer('age')
education_num = cont_layer('education_num')
capital_gain = cont_layer('capital_gain')
capital_loss = cont_layer('capital_loss')
hours_per_week = cont_layer('hours_per_week')

age_buckets = tf.contrib.layers.bucketized_column(age, boundaries = [
    18,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65
])

education_x_occupation = tf.contrib.layers.crossed_column([
        education,
        occupation
    ],
    hash_bucket_size=int(1e4)
)

age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column([
        age_buckets,
        education,
        occupation
    ],
    hash_bucket_size = int(1e6)
)

model_dir = 'tmp/census'

m = tf.contrib.learn.LinearClassifier(feature_columns=[
        gender,
        native_country,
        education,
        occupation,
        workclass,
        marital_status,
        race,
        age_buckets,
        education_x_occupation,
        age_buckets_x_education_x_occupation
    ],
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=1.0,
        l2_regularization_strength=1.0
    ),
    model_dir=model_dir)

for i in range(100):
    m.fit(
        input_fn=train_input_fn,
        steps=200
    )

results = m.evaluate(input_fn=test_input_fn, steps=1)
for key in sorted(results):
    print('%s: %s' % (key, results[key]))

train_file.close()
test_file.close()
