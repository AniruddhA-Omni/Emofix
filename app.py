from os import environ as env
from urllib.parse import quote_plus, urlencode
from chat import get_response
from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, url_for,request, jsonify
import pandas as pd
import numpy as np
import pickle
import json

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)
f = open('Activities_mentalhealth.json')
hdata = json.load(f)

app = Flask(__name__)    
app.secret_key = env.get("APP_SECRET_KEY")

oauth = OAuth(app)

oauth.register(
        "auth0",
        client_id=env.get("AUTH0_CLIENT_ID"),
        client_secret=env.get("AUTH0_CLIENT_SECRET"),
        client_kwargs={
            "scope": "openid profile email",
        },
        server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration',
    )

############################################################################### Auth #############
    # Controllers API
@app.route("/")
def home():
        return render_template(
            "home.html",
            session=session.get("user"), indent=4)
        
    ##pretty=json.dumps(session.get("user")

@app.route("/callback", methods=["GET", "POST"])
def callback():
        token = oauth.auth0.authorize_access_token()
        session["user"] = token
        return redirect("/")


@app.route("/login")
def login():
        return oauth.auth0.authorize_redirect(
            redirect_uri=url_for("callback", _external=True)
        )


@app.route("/logout")
def logout():
        session.clear()
        return redirect(
            "https://"
            + env.get("AUTH0_DOMAIN")
            + "/v2/logout?"
            + urlencode(
                {
                    "returnTo": url_for("home", _external=True),
                    "client_id": env.get("AUTH0_CLIENT_ID"),
                },
                quote_via=quote_plus,
            )
        )
############################################################################### Chat ############# 

@app.route('/predictb', methods=['GET', 'POST'])
def predictb(): 
    if request.method == 'POST':
        text = request.get_json().get("message")
        response = get_response(text)
        message = {'answer': response}
        return jsonify(message)
############################################################################### views #############
model = pickle.load(open('rf_model.pkl', 'rb'))
cols = ['Age', 'Gender', 'self_employed', 'family_history', 'work_interfere','benefits',
       'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
       'mental_health_consequence', 'phys_health_consequence', 'coworkers',
       'supervisor', 'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence', 'age_range']

@app.route('/templates/form.html')    ### form page
def nav():
    return render_template('form.html')


@app.route('/templates/Result.html')
def res():
    return render_template('Result.html')


@app.route('/templates/about.html')
def about():
    return render_template('about.html')


@app.route('/templates/description.html')
def description():
    return render_template('description.html')

@app.route('/templates/activity.html')
def activity():
    return render_template('activity.html')


@app.route('/predict', methods=['POST'])
def form_get():
    peru = request.form['name']
    data1 = int(request.form['Age'])
    data2 = int(request.form['Gender'])
    data3 = int(request.form['self_employed'])
    data4 = int(request.form['family_history'])
    data5 = int(request.form['work_interfere'])
    data6 = int(request.form['benefits'])
    data7 = int(request.form['care_options'])
    data8 = int(request.form['wellness_program'])
    data9 = int(request.form['seek_help'])
    data10 = int(request.form['anonymity'])
    data11 = int(request.form['leave'])
    data12 = int(request.form['mental_health_consequence'])
    data13 = int(request.form['phys_health_consequence'])
    data14 = int(request.form['coworkers'])
    data15 = int(request.form['supervisor'])
    data16 = int(request.form['mental_health_interview'])
    data17 = int(request.form['phys_health_interview'])
    data18 = int(request.form['mental_vs_physical'])
    data19 = int(request.form['obs_consequence'])
    data20 = int(request.form['age_range'])

    arr = np.array([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13,
                    data14, data15, data16, data17, data18, data19, data20])
    data_unseen = pd.DataFrame([arr], columns=cols)
    prediction = model.predict(data_unseen.head(1))[0]
  
    return render_template('result.html', data=prediction, name=peru, hdata=hdata)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=env.get("PORT", 3000),debug=True)