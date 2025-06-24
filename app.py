# resume_classifier.py
import os
import sys
import subprocess
import time
import gradio as gr
import pdfplumber
from transformers import pipeline
import torch
import modal

# Install dependencies only in Colab
if 'google.colab' in sys.modules:
    subprocess.run(["pip", "uninstall", "pyngrok", "-y"], capture_output=True, text=True)
    subprocess.run(["pip", "install", "gradio", "pyngrok", "pdfplumber", "transformers", "torch", "-q"], capture_output=True, text=True)
    subprocess.run(["pip", "install", "--upgrade", "pyngrok", "-q"], capture_output=True, text=True)
    from pyngrok import ngrok, conf

# Define Modal app and image
app = modal.App("resume-classifier")
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "gradio>=4.0.0",
    "pdfplumber>=0.10.0",
    "transformers>=4.30.0",
    "torch>=2.0.0",
)

# Cache models globally to avoid reloading
classifier = None
generator = None

def load_models():
    global classifier, generator
    if classifier is None or generator is None:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
        generator = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)
    return classifier, generator

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
    
    # Load models
    classifier, generator = load_models()
    
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

# Define Gradio interface for Modal
@app.function(
    image=web_image,
    min_containers=1,
    scaledown_window=60 * 20,
    max_containers=1,
    mounts=[modal.Mount.from_local_file(__file__, remote_path="/app/resume_classifier.py")],
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    demo = gr.Blocks(title="Resume Classifier + AI Feedback Generator")
    with demo:
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
    
    demo.queue(max_size=5)  # Enable queue for multiple requests
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")

# Colab testing with ngrok
if __name__ == "__main__" and 'google.colab' in sys.modules:
    os.system("rm -rf ~/.ngrok2")
    os.system("killall ngrok")
    os.system("pkill -f gradio")
    try:
        from pyngrok import ngrok, conf
        ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN", "2yvAIiCVW3z38KDEJgf7kDey60b_74XqMSQxyDjymDtuKofc5"))
        ngrok_process = ngrok.get_ngrok_process()
        ngrok.kill()
        public_url = ngrok.connect(7860, bind_tls=True)
        print(f"Gradio app is running at: {public_url}")
        
        # Define Gradio interface for Colab
        demo = gr.Blocks(title="Resume Classifier + AI Feedback Generator")
        with demo:
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
        
        demo.launch(server_name="0.0.0.0", server_port=7860, quiet=True)
    except Exception as e:
        print(f"Error setting up ngrok tunnel or Gradio server: {str(e)}")
        print("Verify your v2 authtoken at https://dashboard.ngrok.com/get-started/your-authtoken")
        sys.exit(1)
