from flask import Flask, render_template

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
    return render_template('kmeans.html')

if __name__ == "__main__":
    app.run(debug=True)