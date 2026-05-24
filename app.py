from flask import Flask, render_template, request
from kmeans_model import get_kmeans_results

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/crispml")
def crispml():
    return render_template("crispml.html")


@app.route("/business-understanding")
def business_understanding():
    return render_template("business_understanding.html")


@app.route("/data-understanding")
def data_understanding():
    return render_template("data_understanding.html")


@app.route("/data-engineering")
def data_engineering():
    return render_template("data_engineering.html")

@app.route("/dbscan-model")
def dbscan_model():
    return render_template("dbscan_model.html")

@app.route("/PCA-Model")
def PCA_Model():
    return render_template("PCA_Model.html")

@app.route('/kmeans')
def kmeans():
    results = get_kmeans_results()
    return render_template('kmeans.html', results=results)

@app.route("/dbscan-evaluation")
def dbscan_evaluation():
    return render_template("dbscan_evaluation.html")

@app.route("/pca-evaluation")
def pca_evaluation():
    return render_template("PCA_evaluation.html")

@app.route("/kmeans-evaluation")
def kmeans_evaluation():
    # Datos simulados para que la vista cargue sin errores
    # Luego debes conectar esto con tu módulo de agrupamiento real
    mock_results = {
        "n_total": 1000, 
        "n_train": 800,
        "n_val": 200,
        "k_final": 4,
        "features": ["Subíndice 1", "Subíndice 2", "Subíndice 3"],
        "metrics": {
            "val_silhouette": 0.5432,
            "val_db": 0.8765,
            "train_silhouette": 0.5512,
            "train_db": 0.8654,
            "inertia": 2150.45,
            "n_iter": 12
        },
        "predictions": [
            {"cluster": 0, "entity": "Alcaldía de Cajicá", "dept": "Cundinamarca", "orden": "Territorial", "label": "Madurez Media", "pol06": 0.5, "i82": 0.6, "i18": 0.4},
            {"cluster": 1, "entity": "Ministerio TIC", "dept": "Bogotá", "orden": "Nacional", "label": "Madurez Alta", "pol06": 0.9, "i82": 0.8, "i18": 0.9}
        ]
    }
    
    return render_template("kmeans_evaluation.html", results=mock_results)
 

if __name__ == "__main__":
    app.run(debug=True)
