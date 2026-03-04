# Phishing Website Detection using Machine Learning

---

## Problem Defintion:

Phishing is a cyberattack technique in which malicious actors create fraudulent websites that closely resemble legitimate ones to deceive users into revealing sensitive information such as login credentials, financial details, and personal data.

With the rapid growth of online services, phishing attacks have become one of the most significant cybersecurity threats worldwide. Manual detection methods are inefficient, slow, and not scalable. Therefore, there is a strong need for an automated, intelligent detection system capable of identifying phishing websites accurately and in real time.

--- 

## Objective:

The objective of this project is to develop a robust Machine Learning classification system that can accurately predict whether a website is:

- Legitimate

- Phishing

This is formulated as a binary classification problem, where the model learns patterns from structured website feature data and predicts the website status.

---

## Evaluation Criteria;

Since phishing detection is typically an imbalanced classification problem, relying solely on accuracy can be misleading.

Therefore, this project prioritizes:

1. Precision

2. Recall

3. F1 Score (Primary Metric)

F1 Score is used as the primary evaluation metric because it balances Precision and Recall, which is critical in minimizing both:

- False Negatives (dangerous phishing sites not detected)

- False Positives (legitimate websites incorrectly flagged)