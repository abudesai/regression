import os

# root volume
volume_path = './ml_vol/'

# subdirs
logs_path = os.path.join(volume_path, 'logs')
model_path = os.path.join(volume_path, 'model')
output_path = os.path.join(volume_path, 'output')
data_path = os.path.join(volume_path, 'data')

# tain and test data paths
train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')

# ratings file names and paths
train_ratings_fname = 'ratings_train.csv'
test_ratings_fname = 'ratings_test.csv'
train_ratings_fpath = os.path.join(train_data_path, train_ratings_fname)
test_ratings_fpath = os.path.join(test_data_path, test_ratings_fname)