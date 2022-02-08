import algorithm.hpo as hpo
import algorithm.training as train
import algorithm.predictions as predict
import algorithm.scoring as score

'''
ailerons
heart_disease
auto_prices
house_prices
abalone
white_wine
red_wine
computer_activity
'''


dataset = 'house_prices'

datasets_root = f'./../../../datasets/{dataset}'


data_schema_path = f'{datasets_root}/schema/{dataset}_schema.json'
train_data_path = f'{datasets_root}/processed/train/{dataset}_train.csv'
test_data_path = f'{datasets_root}/processed/test/{dataset}_test.csv'


model_name = 'ElasticNet'

model_path = f'{datasets_root}/model/{model_name}/'
output_path = f'{datasets_root}/output/{model_name}/'


# hpo.run_hpo(train_data_path, output_path)

train.run_training( train_data_path, data_schema_path, model_path, logs_path ='./')
predict.run_predictions( test_data_path, model_path, output_path)
score.score_predictions(output_path, data_schema_path)