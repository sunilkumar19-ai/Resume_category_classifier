import streamlit as st
import pickle
import re
import string
from nltk.tokenize import sent_tokenize

# Load pickled models
count = pickle.load(open('vectamir.pkl', 'rb'))
clf = pickle.load(open('modalamir.pkl', 'rb'))

# Preprocess text function
def preprocess_text(text):
    # Remove HTML tags
    clean_text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    clean_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                        clean_text)
    # Remove non-printable characters
    clean_text = ''.join(char for char in clean_text if char in string.printable)
    # Remove unwanted symbols and escape sequences
    clean_text = clean_text.encode('ascii', 'ignore').decode('utf-8')
    clean_text = clean_text.replace("\r\n", "")

    # Tokenize into sentences
    sentences = sent_tokenize(clean_text)

    # Process each sentence
    processed_sentences = []
    for sentence in sentences:
        # Remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        sentence = sentence.lower()

        # Append the processed sentence to the list
        processed_sentences.append(sentence)

    # Join the processed sentences back into a single string
    clean_text = ' '.join(processed_sentences)

    return clean_text


# Streamlit App
st.title("Resume Screening App")
uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

if uploaded_file is not None:
    try:
        resume_bytes = uploaded_file.read()
        resume_text = resume_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # If UTF-8 decoding fails, try decoding with 'latin-1'
        resume_text = resume_bytes.decode('latin-1')

    cleaned_resume = preprocess_text(resume_text)
    input_features = count.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]

    st.write("prediction_id", prediction_id)

    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }

    category_name = category_mapping.get(prediction_id, "Unknown")
    st.write("Predicted Category:", category_name)

    # Predict probabilities for each class
    class_probabilities = clf.predict_proba(input_features)[0]

    # Print probabilities for each class
    for class_id, probability in enumerate(class_probabilities):
        category_name = category_mapping.get(class_id, "Unknown")
        st.write(f"Probability for {category_name}: {probability}")

# st.write("Input Features:", input_features)
