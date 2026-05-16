from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/pipeline")
def pipeline():
    return render_template("pipeline.html")

@app.route("/datasets")
def datasets():
    return render_template("datasets.html")

@app.route("/resultados")
def resultados():
    return render_template("resultados.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)