from params import *

def test_model(model,history):
    loss_and_metrics=model.evaluate(gte, batch_size=16)

    print('Erreur de la base de données TEST',loss_and_metrics[0])
    print('Taux de reconnaisance',loss_and_metrics[1])