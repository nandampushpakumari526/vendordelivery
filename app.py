from flask import Flask, render_template, request, redirect, session
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import joblib
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "majorprojectsecurekey"

MODEL_PATH = "model.pkl"

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS vendor_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_time REAL,
        order_quantity REAL,
        vendor_rating REAL,
        distance_km REAL,
        past_delays REAL,
        season_peak REAL,
        delivered_on_time INTEGER
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- TRAIN MODEL ----------------
def train_model():
    conn = sqlite3.connect("database.db")
    df = pd.read_sql_query("SELECT * FROM vendor_data", conn)
    conn.close()

    if len(df) < 10:
        return None

    df = df.fillna(0)

    # Check class balance
    class_counts = df["delivered_on_time"].value_counts()
    if len(class_counts) < 2:
        return None

    X = df.drop(["id","delivered_on_time"], axis=1)
    y = df["delivered_on_time"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])

    joblib.dump(model, MODEL_PATH)

    with open("model_accuracy.txt", "w") as f:
        f.write(str(round(accuracy * 100, 2)))

    os.makedirs("static/charts", exist_ok=True)

    # ---------------- Confusion Matrix ----------------
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0,1], ["Delayed","On-Time"])
    plt.yticks([0,1], ["Delayed","On-Time"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i][j], ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig("static/charts/confusion_matrix.png")
    plt.close()

    # ---------------- Feature Importance ----------------
    importances = model.feature_importances_
    feature_names = X.columns

    plt.figure(figsize=(8,5))
    bars = plt.bar(feature_names, importances)
    plt.xticks(rotation=45)
    plt.ylabel("Importance Score")
    plt.title("Feature Importance")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height,
                 f"{height:.3f}",
                 ha='center',
                 va='bottom')

    plt.tight_layout()
    plt.savefig("static/charts/importance.png")
    plt.close()

    # ---------------- ROC Curve ----------------
    if len(set(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("static/charts/roc.png")
        plt.close()

# ---------------- SMART SAMPLE DATA GENERATOR ----------------
@app.route("/generate")
def generate_data():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    # Check existing record count
    cur.execute("SELECT COUNT(*) FROM vendor_data")
    total_records = cur.fetchone()[0]

    # ---------------- FIRST GENERATION (Balanced) ----------------
    if total_records == 0:
        # 10 On-Time
        for _ in range(10):
            lead = random.randint(3, 10)
            qty = random.randint(50, 200)
            rating = random.randint(3, 5)
            dist = random.randint(20, 200)
            past = random.randint(0, 2)
            season = 0
            delivered = 1

            cur.execute("""
            INSERT INTO vendor_data
            (lead_time,order_quantity,vendor_rating,distance_km,past_delays,season_peak,delivered_on_time)
            VALUES (?,?,?,?,?,?,?)
            """,(lead,qty,rating,dist,past,season,delivered))

        # 10 Delayed
        for _ in range(10):
            lead = random.randint(20, 35)
            qty = random.randint(300, 600)
            rating = random.randint(1, 2)
            dist = random.randint(400, 900)
            past = random.randint(5, 10)
            season = 1
            delivered = 0

            cur.execute("""
            INSERT INTO vendor_data
            (lead_time,order_quantity,vendor_rating,distance_km,past_delays,season_peak,delivered_on_time)
            VALUES (?,?,?,?,?,?,?)
            """,(lead,qty,rating,dist,past,season,delivered))

    # ---------------- SUBSEQUENT GENERATIONS (Fully Random) ----------------
    else:
        for _ in range(20):
            lead = random.randint(3, 40)
            qty = random.randint(50, 700)
            rating = random.randint(1, 5)
            dist = random.randint(10, 1000)
            past = random.randint(0, 12)
            season = random.randint(0, 1)

            # Smart logic-based outcome (not fixed)
            score = (lead * 0.25) + (past * 0.35) - (rating * 1.5)
            delivered = 0 if score > 8 else 1

            cur.execute("""
            INSERT INTO vendor_data
            (lead_time,order_quantity,vendor_rating,distance_km,past_delays,season_peak,delivered_on_time)
            VALUES (?,?,?,?,?,?,?)
            """,(lead,qty,rating,dist,past,season,delivered))

    conn.commit()
    conn.close()

    return redirect("/dashboard")

# ---------------- CLEAR DATA ----------------
@app.route("/clear")
def clear_data():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM vendor_data")
    conn.commit()
    conn.close()

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    if os.path.exists("model_accuracy.txt"):
        os.remove("model_accuracy.txt")

    charts_path = "static/charts"
    if os.path.exists(charts_path):
        for file in os.listdir(charts_path):
            file_path = os.path.join(charts_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return redirect("/dashboard")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    conn = sqlite3.connect("database.db")
    df = pd.read_sql_query("SELECT * FROM vendor_data", conn)
    conn.close()

    total = len(df)
    on_time = int(df["delivered_on_time"].sum()) if total > 0 else 0
    delayed = total - on_time

    accuracy = None
    if os.path.exists("model_accuracy.txt"):
        with open("model_accuracy.txt","r") as f:
            accuracy = f.read()

    if total > 0:
        os.makedirs("static/charts", exist_ok=True)
        plt.figure()
        plt.pie([on_time, delayed], labels=["On-Time","Delayed"], autopct="%1.1f%%")
        plt.title("Delivery Distribution")
        plt.savefig("static/charts/performance.png")
        plt.close()

    return render_template("dashboard.html",
                           total=total,
                           on_time=on_time,
                           delayed=delayed,
                           accuracy=accuracy)

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username,password))
        user = cur.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect("/dashboard")
        else:
            return "Invalid Credentials"

    return render_template("login.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET","POST"])
def register():
    error = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        existing_user = cur.fetchone()

        if existing_user:
            error = "Username already exists! Please choose another."
        else:
            cur.execute("INSERT INTO users (username,password) VALUES (?,?)",
                        (username,password))
            conn.commit()
            conn.close()
            return redirect("/")

        conn.close()

    return render_template("register.html", error=error)

# ---------------- RETRAIN ----------------
@app.route("/retrain")
def retrain():
    train_model()
    return redirect("/dashboard")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


@app.route("/add", methods=["GET", "POST"])
def add():
    if request.method == "POST":

        lead_time = float(request.form["lead_time"])
        order_quantity = float(request.form["order_quantity"])
        vendor_rating = float(request.form["vendor_rating"])
        distance_km = float(request.form["distance_km"])
        past_delays = float(request.form["past_delays"])
        season_peak = float(request.form["season_peak"])
        delivered_on_time = int(request.form["delivered_on_time"])

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO vendor_data
            (lead_time, order_quantity, vendor_rating,
             distance_km, past_delays, season_peak, delivered_on_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            lead_time, order_quantity, vendor_rating,
            distance_km, past_delays, season_peak, delivered_on_time
        ))

        conn.commit()
        conn.close()

        return redirect("/dashboard")

    return render_template("add.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None

    if request.method == "POST":

        lead_time = float(request.form["lead_time"])
        order_quantity = float(request.form["order_quantity"])
        vendor_rating = float(request.form["vendor_rating"])
        distance_km = float(request.form["distance_km"])
        past_delays = float(request.form["past_delays"])
        season_peak = float(request.form["season_peak"])

        if not os.path.exists(MODEL_PATH):
            train_model()

        if not os.path.exists(MODEL_PATH):
            return "Please add at least 10 records with both delivery classes and train the model."

        model = joblib.load(MODEL_PATH)

        data = [[
            lead_time,
            order_quantity,
            vendor_rating,
            distance_km,
            past_delays,
            season_peak
        ]]

        result = model.predict(data)[0]

        if result == 1:
            prediction = "On-Time Delivery"
        else:
            prediction = "Delayed Delivery"

    return render_template("predict.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)