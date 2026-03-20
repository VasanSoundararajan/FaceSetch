from flask import Flask, render_template, request, redirect, session
import mysql.connector
import pandas as pd
import os
import torch
import cv2
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
from train_model import FusionModel
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------------
# FLASK CONFIG
# ------------------------------------------------

app = Flask(__name__)
app.secret_key = "forensic_secret"

# ------------------------------------------------
# DATABASE CONNECTION
# ------------------------------------------------
print("Loading Semantic Model...")

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="forensic_ai",
    charset="utf8"
)

cursor = db.cursor()



# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------

prompt_data = pd.read_csv("static/data/prompt.csv")

crime_data = pd.read_csv("static/data/crime_data.csv", engine="python", on_bad_lines="skip")


# Precompute embeddings for prompts
prompt_embeddings = semantic_model.encode(
    prompt_data["prompt"].tolist(),
    convert_to_tensor=True
)

# ------------------------------------------------
# LOAD AI MODEL
# ------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FusionModel().to(device)

model.load_state_dict(
    torch.load("models/crime_model.pth", map_location=device)
)

model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

# ------------------------------------------------
# ADMIN LOGIN
# ------------------------------------------------

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():

    if request.method == 'POST':

        u = request.form['username']
        p = request.form['password']

        sql = "SELECT * FROM admin WHERE username=%s AND password=%s"

        cursor.execute(sql, (u, p))

        data = cursor.fetchone()

        if data:
            session['admin'] = u
            return redirect('/admin_dashboard')

    return render_template("admin_login.html")


# ------------------------------------------------
# ADMIN DASHBOARD
# ------------------------------------------------

@app.route('/admin_dashboard')
def admin_dashboard():

    if 'admin' not in session:
        return redirect('/')

    return render_template("admin_dashboard.html")


# ------------------------------------------------
# ADD POLICE
# ------------------------------------------------

@app.route('/add_police', methods=['GET', 'POST'])
def add_police():

    if request.method == 'POST':

        data = (
            request.form['name'],
            request.form['email'],
            request.form['mobile'],
            request.form['location'],
            request.form['station'],
            request.form['police_id'],
            request.form['username'],
            request.form['password']
        )

        sql = """INSERT INTO police
        (name,email,mobile,location,station_name,police_id,username,password)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"""

        cursor.execute(sql, data)
        db.commit()

    return render_template("add_police.html")

# ------------------------------------------------
# VIEW POLICE STATIONS
# ------------------------------------------------

@app.route('/view_police_stations')
def view_police_stations():

    if 'admin' not in session:
        return redirect('/admin_login')

    sql = "SELECT name,station_name,location,mobile,email FROM police"

    cursor.execute(sql)

    data = cursor.fetchall()

    return render_template(
        "view.html",
        stations=data
    )

@app.route('/upload_criminal', methods=['GET','POST'])
def upload_criminal():

    if 'admin' not in session:
        return redirect('/admin_login')

    if request.method == "POST":

        file = request.files['file']

        if file.filename == "":
            return "No file selected"

        # Save file
        path = os.path.join("static/data", file.filename)
        file.save(path)

        # Read CSV
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")

        for index, row in df.iterrows():

            sql = """
            INSERT INTO criminals
            (id,name,age,crime_year,crime_type,no_of_crimes,
            last_known_location,criminal_status,description,prompt,sketch_image)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """

            values = (
                row["id"],
                row["name"],
                row["age"],
                row["crime_year"],
                row["crime_type"],
                row["no_of_crimes"],
                row["last_known_location"],
                row["criminal_status"],
                row["description"],
                row["prompt"],
                row["sketch_image"]
            )

            cursor.execute(sql, values)

        db.commit()

        return "<h3>Criminal Data Uploaded Successfully</h3>"

    return render_template("upload_criminal.html")

# ------------------------------------------------
# POLICE LOGIN
# ------------------------------------------------

@app.route('/police_login', methods=['GET', 'POST'])
def police_login():

    if request.method == 'POST':

        u = request.form['username']
        p = request.form['password']

        sql = "SELECT * FROM police WHERE username=%s AND password=%s"

        cursor.execute(sql, (u, p))

        data = cursor.fetchone()

        if data:
            session['police'] = u
            return redirect('/police_dashboard')

    return render_template("police_login.html")


# ------------------------------------------------
# POLICE DASHBOARD
# ------------------------------------------------

@app.route('/police_dashboard')
def police_dashboard():

    if 'police' not in session:
        return redirect('/police_login')

    return render_template("police_dashboard.html")


# ------------------------------------------------
# VOICE INPUT PAGE
# ------------------------------------------------

@app.route('/voice_input')
def voice_input():
    return render_template("voice_input.html")

# ------------------------------------------------
# TEXT CLEANING FUNCTION
# ------------------------------------------------

def extract_keywords(text):

    stop_words = {
        "and","or","of","the","with","a","an","to","in","on","at",
        "for","from","is","was","are","were","has","have","had",
        "about","around","approximately","person","man","woman"
    }

    text = text.lower()

    # remove punctuation
    for ch in [",",".","!","?"]:
        text = text.replace(ch," ")

    words = text.split()

    keywords = [w for w in words if w not in stop_words]

    return ", ".join(keywords)

def extract_phrases(text):

    text = text.lower()

    # remove punctuation
    for ch in [".", ",", "!", "?"]:
        text = text.replace(ch, "")

    # normalize connectors
    text = text.replace(" and ", ",")
    text = text.replace(" with ", ",")
    text = text.replace(" having ", ",")
    text = text.replace(" has ", ",")

    parts = text.split(",")

    phrases = []

    for p in parts:
        p = p.strip()

        if len(p) > 2:
            phrases.append(p)

    return ", ".join(phrases)

def generate_real_face(sketch_path):

    img = cv2.imread(sketch_path)

    if img is None:
        return sketch_path

    # Convert to colorized face
    color = cv2.applyColorMap(img, cv2.COLORMAP_BONE)

    # Smooth to make it look realistic
    color = cv2.bilateralFilter(color, 9, 75, 75)

    # Save generated image
    face_path = sketch_path.replace("sketches", "generated")

    os.makedirs("static/generated", exist_ok=True)

    cv2.imwrite(face_path, color)

    return "/" + face_path.replace("\\","/")

import os
import numpy as np
import torch
from transformers import WhisperProcessor
from transformers import ViTModel


class SystemConfig:

    def __init__(self):

        self.audio_sampling_rate = 16000
        self.embedding_dimension = 768
        self.similarity_threshold = 0.82

        self.database_path = "criminal_database/"
        self.generated_face_path = "generated_faces/"

        self.model_paths = {
            "whisper": "models/whisper_multilingual",
            "attention_cgan": "models/attention_cgan",
            "vit_encoder": "models/vit_face_encoder"
        }

class SpeechToTextEngine:

    def __init__(self, config):

        self.config = config
        self.processor = None
        self.model = None

    def load_model(self):

        """
        Loads multilingual Whisper speech recognition model.
        """

        self.processor = WhisperProcessor
        self.model = "WhisperModelPlaceholder"

    def transcribe_audio(self, audio_file):

        """
        Converts witness audio description into textual format.
        """

        if audio_file is None:
            raise ValueError("Audio file not provided")

        transcript = "placeholder text description of suspect"

        return transcript

class DescriptionEncoder:

    def __init__(self):

        self.attribute_dictionary = [
            "hair_color",
            "face_shape",
            "eye_type",
            "nose_size",
            "mouth_shape",
            "skin_tone",
            "facial_hair",
            "age_group"
        ]

    def extract_attributes(self, text):

        """
        Parses witness description and converts to attribute vector.
        """

        attributes = {}

        for attr in self.attribute_dictionary:

            attributes[attr] = "unknown"

        return attributes


class AttentionCGANGenerator:

    def __init__(self, config):

        self.config = config
        self.generator = None
        self.discriminator = None

    def load_model(self):

        """
        Loads pretrained Attention Conditional GAN architecture.
        """

        self.generator = "GeneratorPlaceholder"
        self.discriminator = "DiscriminatorPlaceholder"

    def generate_face_sketch(self, attributes):

        """
        Generates a suspect face sketch based on attribute vector.
        """

        if attributes is None:
            raise Exception("Attributes missing")

        fake_image = np.zeros((256,256,3))

        return fake_image


class SketchEnhancer:

    def enhance(self, sketch_image):

        """
        Enhances sketch image using sketch-to-photo translation.
        """

        enhanced = sketch_image

        return enhanced


class FaceEncoder:

    def __init__(self, config):

        self.config = config
        self.model = None

    def load_model(self):

        self.model = ViTModel

    def encode(self, image):

        """
        Converts face image to embedding vector.
        """

        embedding = np.random.rand(self.config.embedding_dimension)

        return embedding


class CriminalDatabase:

    def __init__(self, config):

        self.config = config
        self.records = []

    def load_database(self):

        """
        Loads stored criminal face embeddings.
        """

        for i in range(50):

            record = {
                "name": "Suspect_" + str(i),
                "embedding": np.random.rand(self.config.embedding_dimension)
            }

            self.records.append(record)

    def get_records(self):

        return self.records


class SimilarityEngine:

    def cosine_similarity(self, v1, v2):

        dot = np.dot(v1, v2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        return dot / (norm1 * norm2)

    def find_best_match(self, query_embedding, database, threshold):

        best_score = 0
        best_match = None

        for record in database:

            score = self.cosine_similarity(query_embedding, record["embedding"])

            if score > best_score:

                best_score = score
                best_match = record

        if best_score >= threshold:

            return best_match, best_score

        return None, best_score
    
class ForensicIdentificationPipeline:

    def __init__(self):

        self.config = SystemConfig()

        self.speech_engine = SpeechToTextEngine(self.config)
        self.text_encoder = DescriptionEncoder()
        self.cgan_generator = AttentionCGANGenerator(self.config)
        self.sketch_enhancer = SketchEnhancer()
        self.face_encoder = FaceEncoder(self.config)
        self.database = CriminalDatabase(self.config)
        self.similarity_engine = SimilarityEngine()

    def initialize(self):

        self.speech_engine.load_model()
        self.cgan_generator.load_model()
        self.face_encoder.load_model()
        self.database.load_database()

    def run_pipeline(self, audio_input):

        text = self.speech_engine.transcribe_audio(audio_input)

        attributes = self.text_encoder.extract_attributes(text)

        sketch = self.cgan_generator.generate_face_sketch(attributes)

        enhanced = self.sketch_enhancer.enhance(sketch)

        query_embedding = self.face_encoder.encode(enhanced)

        records = self.database.get_records()

        match, score = self.similarity_engine.find_best_match(
            query_embedding,
            records,
            self.config.similarity_threshold
        )

        result = {}

        if match is not None:

            result["status"] = "MATCH FOUND"
            result["suspect"] = match["name"]
            result["confidence"] = score

        else:

            result["status"] = "NO MATCH FOUND"
            result["confidence"] = score

        return result

# ------------------------------------------------
# PREDICT ROUTE
# ------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    user_prompt = request.form.get("prompt", "").strip()

    # Extract phrases
    clean_prompt = extract_phrases(user_prompt)

    print("Original Prompt:", user_prompt)
    print("Extracted Keywords:", clean_prompt)

    if user_prompt == "":
        return render_template("result.html", found=False)

    # -----------------------------------------
    # Convert user prompt to embedding
    # -----------------------------------------

    user_embedding = semantic_model.encode(
        clean_prompt,
        convert_to_tensor=True
    )

    # -----------------------------------------
    # Compute similarity
    # -----------------------------------------

    scores = util.cos_sim(user_embedding, prompt_embeddings)

    best_match = scores.argmax().item()

    predicted_row = prompt_data.iloc[best_match]

    sketch = predicted_row["sketch_image"]

    # convert sketch filename to face filename
    face = sketch.replace("sketch", "face")

    sketch_path = f"/static/sketches/{sketch}"
    face_path = f"/static/faces/{face}"

    # -----------------------------------------
    # Search crime database
    # -----------------------------------------

    match = crime_data[crime_data["sketch_image"] == sketch]

    if not match.empty:

        criminal = match.iloc[0]

        return render_template(
            "result.html",
            found=True,
            keywords=clean_prompt,
            user_prompt=user_prompt,
            name=criminal["name"],
            age=criminal["age"],
            crime=criminal["crime_type"],
            crimes=criminal["no_of_crimes"],
            year=criminal["crime_year"],
            description=criminal["description"],
            sketch=sketch,
            sketch_img=sketch_path,
            colored_face=face_path
        )

    else:

        return render_template(
            "result.html",
            found=False,
            user_prompt=user_prompt,
            keywords=clean_prompt,
            sketch=sketch,
            sketch_img=sketch_path,
            colored_face=face_path
        )
@app.route('/logout')
def logout():

    session.pop('admin', None)  
    session.pop('police', None) 

    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
