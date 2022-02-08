import flask
import algorithm.model_config as cfg
from train import train 
from predict import predict
from score import score
from tune import tune 

app = flask.Flask(__name__)



@app.route("/", methods=["GET"])
def hello():
    return f'Hello - I am {cfg.MODEL_NAME} and I am alive!\n'
     

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    
    # some health check here ...
    health = True

    status = 200 if health else 404
    return flask.Response(response="Ping!\n", status=status, mimetype="application/json")


@app.route("/train", methods=["GET"])
def train_endpoint():
    resp = train()
    status = 200 if resp==0 else 404
    return flask.Response(response="Training completed successfully.", status=status, mimetype="application/json")


@app.route("/predict", methods=["GET"])
def predict_endpoint(): 
    resp = predict()
    if resp == 0: 
        status = 200 
        resp = "Predictions completed successfully."
    else:
        status = 404
    return flask.Response(response=resp, status=status, mimetype="application/json")


@app.route("/score", methods=["GET"])
def score_endpoint():
    resp = score()
    if resp == 0: 
        status = 200 
        resp = "Scoring completed successfully."
    else:
        status = 404
    return flask.Response(response=resp, status=status, mimetype="application/json")


@app.route("/tune", methods=["GET"])
def tune_endpoint():
    resp = tune()
    if resp == 0: 
        status = 200 
        resp = "HPT completed successfully. Check HPT results in output dir."
    else:
        status = 404
    return flask.Response(response=resp, status=status, mimetype="application/json")




if __name__ == "__main__":
    app.run(host= '0.0.0.0',port=3000, debug=True)
    
	