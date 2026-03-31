# Tuberculosis (TB) Detection using CNN and NLP

## Overview
This project is an AI-based system for detecting **Tuberculosis (TB)** using a combination of **Convolutional Neural Networks (CNN)** and **Natural Language Processing (NLP)** techniques. It analyzes chest X-ray images along with textual data (such as reports or symptoms) to improve prediction accuracy.
##  Methodology
- Used **CNN** for image-based TB detection from chest X-rays  
- Applied **NLP techniques** to process textual data such as medical reports or symptoms  
- Combined outputs from both models for better decision-making  
- Performed preprocessing including image resizing, normalization, and text cleaning  
##  Features
- Multi-modal approach (Image + Text)  
- Improved prediction using combined models  
- Automated TB detection system  
- Scalable for healthcare applications  
##  Model Details
- Image Model: CNN (for X-ray classification)  
- Text Model: NLP (for report/symptom analysis)  
- Input: X-ray images + text data  
- Output: TB Positive / Normal  
- Frameworks: TensorFlow / Keras, NLTK / spaCy  
##  Installation

```bash
git clone https://github.com/your-username/tb-detection.git
cd tb-detection
pip install -r requirements.txt
