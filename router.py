"""
router.py - ML-based incident classifier for Synapse-OPS
Uses logistic regression on TF-IDF features for robust classification.
"""

import os
import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import numpy as np
import joblib

MODEL_PATH = "incident_classifier.pkl"


def evaluate_model(pipeline, descriptions, labels):
    """
    Print classification report + average confidence per class
    """
    X_val = descriptions
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)

    print("\n=== MODEL EVALUATION ===")
    print(classification_report(labels, y_pred, zero_division=0))

    # Average confidence per class
    class_confidences = {}
    for label, probs in zip(labels, y_prob):
        conf = np.max(probs)
        class_confidences.setdefault(label, []).append(conf)

    print("=== AVG CONFIDENCE PER CLASS ===")
    for cls in sorted(class_confidences.keys()):
        avg_conf = np.mean(class_confidences[cls])
        print(f"{cls}: {avg_conf:.2f}")


def train_model() -> Pipeline:
    """
    Train a fresh model on sample training data and save it.
    """
    df = pd.read_csv("sample_train_data.csv")
    pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),       # unigrams + bigrams
        stop_words="english",     # remove common stopwords
        max_features=5000,        # keep top 5000 features
        min_df=1                 # include rare words (even if appears once)
    )),
    ("clf", LogisticRegression(
        max_iter=500,             # more iterations for convergence
        class_weight="balanced",  # balance class weights
        C=2.0,                    # lower regularization → better separation
        solver="lbfgs"            # reliable solver for multi-class
    ))
])
    pipeline.fit(df["description"], df["category"])
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[TRAIN] Model trained and saved to {MODEL_PATH}")

    # Evaluate right after training
    evaluate_model(pipeline, df["description"], df["category"])
    return pipeline


# Load model or train new one if missing
if os.path.exists(MODEL_PATH):
    classifier = joblib.load(MODEL_PATH)
    print(f"[LOAD] Incident classifier loaded from {MODEL_PATH}")
else:
    classifier = train_model()


def classify_incident(description: str, threshold: float = 0.3) -> Tuple[str, float]:
    """
    Classify incident using ML model.
    Returns (predicted_category, confidence).
    """
    print(f"--- INSIDE classify_incident --- Received: '{description}'")
    probs = classifier.predict_proba([description])[0]
    classes = classifier.classes_
    max_idx = probs.argmax()
    predicted_category = classes[max_idx]
    confidence = probs[max_idx]

    if confidence < threshold:
        return "UNKNOWN", confidence
    return predicted_category, confidence


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    if args.retrain:
        classifier = train_model()

    # Run quick test cases
    test_cases = [
        "Merchant says orders delayed 35 minutes due to heavy load.",
        "Kitchen queue very long, drivers waiting outside.",
        "Store suddenly went offline during dinner hours.",
        "Merchant cancelled because restaurant shut down for maintenance.",
        "Customer says dessert was not delivered because it's unavailable.",
        "Item unavailable, merchant suggested replacement dish but customer refused.",
        "Driver stuck in heavy rain traffic, ETA 40 mins late.",
        "Driver not responding to calls or chat for 15 minutes.",
        "Noodles spilled in the bag, customer requesting refund.",
        "Food arrived soggy and leaking, packaging compromised.",
        "Customer was charged twice for the same order.",
        "Order failed but amount was debited from card.",
        "Merchant overloaded and kitchen is very slow, customer asking for refund.",
        "Driver breakdown reported and order needs reassignment."
        "Customer reports missing drink from order"
    ]
    ambiguous_test_cases = [
    # Could be backlog or closed
    "Merchant taking too long, might close soon.",
    "Kitchen prep very slow, orders being cancelled by customers.",

    # Could be backlog or driver issue
    "Drivers waiting too long at merchant, complaining about delays.",
    "Merchant is busy and driver is idle outside for 25 minutes.",

    # Could be item out of stock or merchant backlog
    "Customer says order is delayed because ingredients ran out.",
    "Order partially prepared, missing one item still in kitchen.",

    # Could be driver issue or merchant backlog
    "Delivery delayed because merchant prep is slow and driver left location.",
    "Driver reassigned mid-order due to long wait time.",

    # Could be packaging issue or refund policy
    "Customer says box slightly dented but food is fine.",
    "Spillage very minor, customer just asking for discount voucher.",

    # Could be payment issue or merchant closed
    "Order auto-cancelled, amount deducted but merchant was offline.",
    "System shows merchant offline and payment not processed yet.",

    # Mixed signals (hard cases)
    "Customer says order delayed, driver unreachable, food might be cold.",
    "Merchant accepted order but later cancelled due to missing item."
    ]
    hella_ambiguous_cases = [
        # Mixed merchant + driver + time delay
        "Order taking forever, driver has been waiting, maybe kitchen is closed?",
        "Merchant slow and driver unreachable, food probably cold now.",
        "Kitchen backlog suspected, but merchant just came online again.",
        "Driver left pickup point, customer upset about long prep time.",
        "Merchant might cancel order if not prepared soon, driver already reassigned once.",

        # Mixed merchant + item availability
        "Merchant says one item is out of stock, rest will take 20 more minutes.",
        "Half order ready, rest missing, merchant suggested replacement.",
        "Customer agreed to replacement but wants compensation for delay.",

        # Packaging but borderline
        "Container lid slightly loose, no major spill but customer unhappy.",
        "Customer claims packaging not sealed properly, but food intact.",
        "Minor spillage reported but no refund requested yet.",

        # Payment + merchant closed
        "Payment deducted but app shows merchant unavailable.",
        "Merchant was offline briefly, payment took long time to confirm.",
        "Refund still pending, but merchant accepted order later.",

        # Super vague complaints
        "Customer says experience was bad and wants action.",
        "Order not delivered on time, unsure if driver or merchant is at fault.",
        "Customer upset about overall service quality, no details given.",
        "Driver called and said order issue but did not specify what.",
        "Merchant reported a problem but reason not shared with system.",

        # Conflicting signals
        "Driver marked order delivered but customer says food missing.",
        "Merchant confirmed prep but app still says pending.",
        "Payment confirmed twice but order cancelled automatically.",
        "Driver says merchant closed but system shows open."
    ]


    for text in test_cases:
        category, conf = classify_incident(text)
        print(f"[TEST] '{text}' → {category.upper()} (conf={conf:.2f})")
    
    print("\n=== RUNNING AMBIGUOUS TEST CASES ===")
    for text in ambiguous_test_cases:
        category, conf = classify_incident(text)
        print(f"[AMBIG] '{text}' → {category.upper()} (conf={conf:.2f})")

    print("\n=== RUNNING HELLA AMBIGUOUS TEST CASES ===")
    for text in hella_ambiguous_cases:
        category, conf = classify_incident(text)
        print(f"[HELLA-AMBIG] '{text}' → {category.upper()} (conf={conf:.2f})")

