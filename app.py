import os
import sys
import subprocess
import time

# Uninstall pyngrok to ensure clean install
subprocess.run(["pip", "uninstall", "pyngrok", "-y"], capture_output=True, text=True)
# Install required libraries
subprocess.run(["pip", "install", "gradio", "pyngrok", "pdfplumber", "transformers", "torch", "-q"], capture_output=True, text=True)
# Ensure latest pyngrok
subprocess.run(["pip", "install", "--upgrade", "pyngrok", "-q"], capture_output=True, text=True)

# Import libraries
import gradio as gr
import pdfplumber
from transformers import pipeline
import torch
from pyngrok import ngrok, conf

# Clear ngrok cache to prevent binary issues
os.system("rm -rf ~/.ngrok2")

# Initialize ngrok binary by starting a dummy process
try:
    # Set authtoken first to ensure ngrok is configured
    ngrok.set_auth_token("2yvAIiCVW3z38KDEJgf7kDey60b_74XqMSQxyDjymDtuKofc5")
    # Start a temporary ngrok process to trigger binary download
    ngrok_process = ngrok.get_ngrok_process()
    ngrok.kill()  # Stop the temporary process
    print("Ngrok binary initialized successfully")
except Exception as e:
    print(f"Error initializing ngrok binary: {str(e)}")
    print("Verify your v2 authtoken at https://dashboard.ngrok.com/get-started/your-authtoken")
    sys.exit(1)

# Cache models to avoid reloading
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
    generator = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)
    return classifier, generator

# Load models
classifier, generator = load_models()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Main processing function for Gradio
def analyze_resume(pdf_file):
    if not pdf_file:
        return "Please upload a PDF resume.", "", ""

    # Extract text
    resume_text = extract_text_from_pdf(pdf_file)
    if "Error" in resume_text:
        return resume_text, "", ""

    # Preview text (limit to 2000 chars)
    text_preview = resume_text[:2000]

    # Classify job role
    try:
        job_roles = [
            "Software Engineer", "Data Analyst", "Data Scientist",
            "Backend Developer", "UI/UX Designer"
        ]
        result = classifier(resume_text[:1000], job_roles)
        predicted_role = result["labels"][0]
    except Exception as e:
        return text_preview, f"Error classifying role: {str(e)}", ""

    # Generate feedback
    try:
        prompt = f"Provide resume improvement suggestions for a {predicted_role} based on the following:\n{resume_text[:1000]}"
        feedback = generator(prompt, max_length=200)[0]['generated_text']
    except Exception as e:
        return text_preview, predicted_role, f"Error generating feedback: {str(e)}"

    return text_preview, predicted_role, feedback

# Define Gradio interface
with gr.Blocks(title="Resume Classifier + AI Feedback Generator") as demo:
    gr.Markdown("# 📄 Resume Classifier + AI Feedback Generator")
    pdf_input = gr.File(label="Upload your resume (PDF only)", file_types=[".pdf"])
    analyze_button = gr.Button("🔎 Analyze Resume")
    text_preview = gr.Textbox(label="📜 Extracted Resume Preview", lines=10, interactive=False)
    predicted_role = gr.Textbox(label="🎯 Predicted Role", interactive=False)
    feedback = gr.Textbox(label="📝 AI Feedback", lines=5, interactive=False)
    analyze_button.click(
        fn=analyze_resume,
        inputs=pdf_input,
        outputs=[text_preview, predicted_role, feedback]
    )

# Start Gradio server and ngrok tunnel
if __name__ == "__main__":
    # Kill existing ngrok and Gradio processes
    os.system("killall ngrok")
    os.system("pkill -f gradio")

    # Start Gradio server with ngrok
    try:
        public_url = ngrok.connect(7860, bind_tls=True)  # Gradio default port is 7860
        print(f"Gradio app is running at: {public_url}")
        demo.launch(server_name="0.0.0.0", server_port=7860, quiet=True)
    except Exception as e:
        print(f"Error setting up ngrok tunnel or Gradio server: {str(e)}")
        print("Verify your v2 authtoken at https://dashboard.ngrok.com/get-started/your-authtoken")
        sys.exit(1)
