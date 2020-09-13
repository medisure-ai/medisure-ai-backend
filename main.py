from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS

import os
from dotenv import load_dotenv, find_dotenv

from scripts.gpt import GPT, Example, set_openai_key
import openai

from google.cloud import storage

from scripts.vision import parse_table

UPLOAD_FOLDER = os.path.join(".", "uploads")
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CORS_HEADERS"] = "Content-Type"
CORS(app)

load_dotenv(find_dotenv(), override=True)
set_openai_key(os.getenv("GPT_SECRET_KEY", ""))

BUCKET_NAME = os.getenv("GOOGLE_BUCKET_NAME", "")
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/vision", methods=["GET", "POST"])
def parse_pdf():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filename = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filename)
    return parse_table(filename)[0]


vision_GPT = GPT(engine="davinci", temperature=0.5, max_tokens=200)
vision_qa_examples = [
    [
        "What is not included in the out of pocket limit?",
        "Premiums and health cares this plan does not cover are not included. Even though you pay these expenses, they do not count towards the limit",
    ],
    [
        "Do I need a referral to see a specialist?",
        "No. You can see a specialist you choose without permission from this plan.",
    ],
    [
        "Are diagnostic x-rays covered?",
        "In network diagnostics are a maximum of $10 copay while out of network diagnostics are not covered.",
    ],
]


@app.route("/vision/qa", methods=["GET", "POST"])
def question_answer():
    blob = bucket.blob(request.args.get("doc"))
    text = blob.download_as_text()
    vision_GPT.set_premise(text)
    vision_GPT.delete_all_examples()
    for example in vision_qa_examples:
        vision_GPT.add_example(Example(example[0], example[1]))
    prompt = request.data.decode("UTF-8")
    return vision_GPT.get_top_reply(prompt)


@app.route("/vision/summary", methods=["GET", "POST"])
def summarize_doc():
    blob = bucket.blob(request.args.get("doc"))
    prompt = "Could you summarize this in an easy to understand manner?"
    vision_GPT.set_premise(blob.download_as_text())
    vision_GPT.delete_all_examples()
    return vision_GPT.get_top_reply(prompt)


denial_GPT = GPT(engine="davinci", temperature=0.2, max_tokens=200)
denial_GPT.set_premise("Extract the required fields from the letter.")
denial_examples = [
    [
        """
        Aetna
        P.O. Box 3013 Mad Stop U12W Blue Bell, PA 19422-0763

        Powell Gonzalez
        54 Creek Street
        Philadelphia
        PA

        Dear Mr. Powell
        Thank you for your claim application for an Aetna Advantage Plans for Individuals, Families, and the Self-Employed. Your claim was for your condition of having trouble breathing. Based on the documents submitted, this condition was diagnosed on May 10, 2020 and the procedure took play on August 10, 2020 under Dr. Amit Patel of the Philadelphia General Hospital. After careful review of your application, we are unable to offer coverage and have declined the claim for the following reason(s): On review of the test reports submitted, this specific procedure of septoplasty did not meet the standard of being considered medically necessary and was deemed a cosmetic procedure. Furthermore, the provider (Philadelphia General Hospital)is an out-of-network provider- based on the specifics of the Aetna Advantage Plus Plan, only procedures significantly critical to patient health can be covered out-of-network. This information may have come from your application, phone interview or medical records. Medical factors that we did not review this time may be considered.
        """,
        """
        Insurance Plan: Aetna Advantage Plus for Individuals, Families and Self-Employed
        Reason: The reason the claim was denied was that the septoplasty procedure was deemed cosmetic and not medically necessary
        Patient Name: Powell Gonzalez
        State: PA
        Diagnosis Date: May 10, 2020
        Treatment Date: August 10, 2020
        Condition type: Respiratory
        Condition: Trouble breathing
        Supervising Doctor: Dr. Amit Patel
        Hospital: Philadelphia General Hospital
        In-network provider: No
        Treatment/procedure name: Septoplasty
        """,
    ],
    [
        """
        Cigna
        Cigna P.O. Box 8230 Mail, NY, NY 10027

        Rachel Morrison
        75 Palace enclave
        Los Angeles
        CA

        Dear Ms. Morrison,
        This is regarding your recent claim under your Cigna MD PPO.  From the medical documents and forms submitted, your claim application for treatment was considered throughly. Your condition was diagnosed on December 10, 2018 and the gastric bypass treatment started June 10, 2019 under Dr. Mark Henderson of the Good Samaritan Hospital. We regret to inform you that we are unable to cover your claim for this treatment because of the following considerations: Through careful evaluation of your medical history, your condition of diabetes was deemed self-inflicted. Aetna's doctors and in-house medical expertise concluded that based on the test results submitted, despite an early diagnosis, patient behaviour did not change. If you wish to do so, you may submit an appeal.
        """,
        """
        Insurance Plan: Cigna MD PPO
        Reason: The reason the claim was denied was that the diabetes procedure was deemed self-inflicted
        Patient Name: Rachel Morrison
        State: CA
        Diagnosis Date: December 10, 2018
        Treatment Date: June 10, 2019
        Condition type: Diabetes
        Condition: Diabetes
        Supervising Doctor: Dr. Mark Henderson
        Hospital: Philadelphia General Hospital
        In-network provider: Yes
        Treatment/procedure name: Gastric bypass
        """,
    ],
]
for example in denial_examples:
    denial_GPT.add_example(Example(example[0], example[1]))


@app.route("/denial", methods=["GET", "POST"])
def parse_denial():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filename = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filename)
    data = parse_table(filename)[1]
    print(data)
    print()
    text = "[[Letter]]:\nAetna \nP.O. Box 3013 Mad Stop U12W Blue Bell, PA 19422-0763\n\nPowell Gonzalez\n54 Creek Street\nPhiladelphia\nPA\n\nDear Mr. Powell\nThank you for your claim application for an Aetna Advantage Plans for Individuals, Families, and the Self-Employed. Your claim was for your condition of having trouble breathing. Based on the documents submitted, this condition was diagnosed on May 10, 2020 and the procedure took play on August 10, 2020 under Dr. Amit Patel of the Philadelphia General Hospital. After careful review of your application, we are unable to offer coverage and have declined the claim for the following reason(s): On review of the test reports submitted, this specific procedure of septoplasty did not meet the standard of being considered medically necessary and was deemed a cosmetic procedure. Our medical experts did not deem it medically necessary for your situation. This information may have come from your application, phone interview or medical records. Medical factors that we did not review this time may be considered.\n\n[[Summarized Version]]:Insurance Plan: Aetna Advantage Plus for Individuals, Families and Self-Employed, Reason: The reason the claim was denied was that the septoplasty procedure was deemed cosmetic and not medically necessary, Patient Name: Powell Gonzalez, Patient Plan State: PA,  Diagnosis Date: May 10, 2020, Treatment Date: August 10, 2020, Condition type: Respiratory, Condition: Trouble breathing, Supervising Doctor: Dr. Amit Patel, Treatment/procedure name: septoplasty\n\n\n[[Letter]]:\nCigna\nCigna P.O. Box 8230 Mail, NY, NY 10027\n\nRachel Morrison\n75 Palace enclave\nLos Angeles\nCA\n\nDear Ms. Morrison,\nThis is regarding your recent claim under your Cigna MD PPO.  From the medical documents and forms submitted, your claim application for treatment was considered throughly. Your condition was diagnosed on December 10, 2018 and the gastric bypass treatment started June 10, 2019 under Dr. Mark Henderson of the Good Samaritan Hospital. We regret to inform you that we are unable to cover your claim for this treatment because of the following considerations: Through careful evaluation of your medical history, your condition of diabetes was deemed self-inflicted. Aetna's doctors and in-house medical expertise concluded that based on the test results submitted, despite an early diagnosis, patient behaviour did not change .  If you wish to do so, you may submit an appeal.\n\n[[Summarized Version]]:Insurance Plan: Cigna MD PPO, Reason: The reason the claim was denied was that the diabetes procedure was deemed self-inflicted, Patient Name: Rachel Morrison, Patient Plan State: CA, Diagnosis Date: December 10, 2018, Treatment Date: June 10, 2019, Condition type: Diabetes, Condition: Diabetes, Supervising Doctor: Dr. Mark Henderson, Treatment/procedure name: Gastric bypass\n\n"
    text += "[[Letter]]:\n" + data
    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        temperature=0.33,
        max_tokens=125,
        top_p=1,
        stop=["\n\n"],
    )

    response = openai.Completion.create(
        engine="davinci",
        prompt="[[Letter]]:\nAetna \nP.O. Box 3013 Mad Stop U12W Blue Bell, PA 19422-0763\n\nPowell Gonzalez\n54 Creek Street\nPhiladelphia\nPA\n\nDear Mr. Powell\nThank you for your claim application for an Aetna Advantage Plans for Individuals, Families, and the Self-Employed. Your claim was for your condition of having trouble breathing. Based on the documents submitted, this condition was diagnosed on May 10, 2020 and the procedure took play on August 10, 2020 under Dr. Amit Patel of the Philadelphia General Hospital. After careful review of your application, we are unable to offer coverage and have declined the claim for the following reason(s): On review of the test reports submitted, this specific procedure of septoplasty did not meet the standard of being considered medically necessary and was deemed a cosmetic procedure. Our medical experts did not deem it medically necessary for your situation. This information may have come from your application, phone interview or medical records. Medical factors that we did not review this time may be considered.\n\n[[Summarized Version]]:Insurance Plan: Aetna Advantage Plus for Individuals, Families and Self-Employed, Reason: The reason the claim was denied was that the septoplasty procedure was deemed cosmetic and not medically necessary, Patient Name: Powell Gonzalez, Patient Plan State: PA,  Diagnosis Date: May 10, 2020, Treatment Date: August 10, 2020, Condition type: Respiratory, Condition: Trouble breathing, Supervising Doctor: Dr. Amit Patel, Treatment/procedure name: septoplasty\n\n\n[[Letter]]:\nCigna\nCigna P.O. Box 8230 Mail, NY, NY 10027\n\nRachel Morrison\n75 Palace enclave\nLos Angeles\nCA\n\nDear Ms. Morrison,\nThis is regarding your recent claim under your Cigna MD PPO.  From the medical documents and forms submitted, your claim application for treatment was considered throughly. Your condition was diagnosed on December 10, 2018 and the gastric bypass treatment started June 10, 2019 under Dr. Mark Henderson of the Good Samaritan Hospital. We regret to inform you that we are unable to cover your claim for this treatment because of the following considerations: Through careful evaluation of your medical history, your condition of diabetes was deemed self-inflicted. Aetna's doctors and in-house medical expertise concluded that based on the test results submitted, despite an early diagnosis, patient behaviour did not change .  If you wish to do so, you may submit an appeal.\n\n[[Summarized Version]]:Insurance Plan: Cigna MD PPO, Reason: The reason the claim was denied was that the diabetes procedure was deemed self-inflicted, Patient Name: Rachel Morrison, Patient Plan State: CA, Diagnosis Date: December 10, 2018, Treatment Date: June 10, 2019, Condition type: Diabetes, Condition: Diabetes, Supervising Doctor: Dr. Mark Henderson, Treatment/procedure name: Gastric bypass\n\n[[Letter]]:\nAetna\nAetna P.O. Box 3013 Mad Stop U12W Blue Bell, PA 19422-0763\nDear REDACTED\nThank you for applying for an Aetna Advantage Plans for Individuals, Families, and the Self-Employed.\nAfter carefully evaluating your application, we are unable to offer coverage to you at this time.\nCoverage is not guaranteed under an Aetna Advantage Plan for Individuals, Families and the Self-\nEmployed All applicants and/or their dependents are medically underwritten to determine each\nindividual's risk category. We offer various risk categories based on each applicant's medical risk factors.\nAfter careful review of your application, we are unable to offer coverage and have declined the\nfollowing applicant for the following reason(s):\nSeborrheic Keratosis, Seasonal Allergies, Giant Papillary Conjunctivitis, Restless Legs Syndrome,\nHeadaches, Neck Pain and TMJ.\nThis determination is based on review of your medical conditions and associated treatment, which may\ninclude medications. This information may have come from your application, phone interview or\nmedical records.\nYou may want to apply again in the future. If you do, we will review your medical history. Medical\nfactors that we did not review this time may be considered. This means you might be offered a higher\nrate or be declined again.\nAlthough not eligible for an Aetna Advantage Plan for Individuals, Families and the Self-Employed, you\nmay be eligible for coverage under the Texas Health Insurance Risk Pool. Please contact the Texas\nHealth Insurance Risk Pool toll free at 1-888-398-3927 for enrollment and eligibility information. The\nhearing and speech impaired may call toll free at TDD 1-800-735-2989. You can also access information\nregarding this plan at their website http://txhealthpool.org/.\nYou may also contact the Foundation for Health Coverage at 1-800-234-1317 or visit their website\nat www.coverageforall.org.\nYour EFT account will not be debited.\nYou have the right to appeal our decision You may submit a request for an appeal in writing. Your appeal\nmust be received within 180 days of the underwriting decision. Your request must include:\n",
        temperature=0.33,
        max_tokens=125,
        top_p=1,
        stop=["\n\n"],
    )
    print(response)

    return data


summarize_GPT = GPT(engine="davinci", temperature=0.5, max_tokens=100)
summarize_GPT.set_premise("This bots answers questions about medical insurance.")
summarize_examples = [
    [
        "Q: What is cancer?",
        "A: Cancer can have many symptoms including fatigue, pain, and tissue masses. Cancer develops when the body's normal control mechanism stops working. There are many kinds of cancer.",
    ],
    [
        "Q: What is diabetes?",
        "A: Diabetes is a disease where your blood sugar levels are too high. With type 1 diabetes, your body does not make insulin, a hormone that helps utilize glucose.",
    ],
    [
        "Q: What is a copay?",
        "A: A copay is a fixed out-of-pocket amount paid by an insured for covered services. A copay is often specified in your insurance policy.",
    ],
]
for example in summarize_examples:
    summarize_GPT.add_example(Example(example[0], example[1]))


@app.route("/summary", methods=["GET", "POST"])
def gpt3():
    prompt = request.data.decode("UTF-8")
    print(prompt)
    out = summarize_GPT.get_top_reply(prompt)
    return out


if __name__ == "__main__":
    app.run(host="0.0.0.0")
