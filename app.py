import gradio as gr
import subprocess
import os

# This function acts as a bridge to the project's original demo.py
def generate_gesture(audio_path, text):
    if not audio_path:
        return "Please upload an audio file."
    
    # We will call the project's demo script via command line
    # Using the Shortcut model config as it's the fastest
    config_path = "configs_new/meanflow_rvqvae_128.yaml"
    
    # Construct the command
    # Note: Adjust the flags if the repo uses different ones (e.g., -a instead of --audio)
    command = [
        "python", "demo.py",
        "--config", config_path,
        "--audio", audio_path,
        "--text", text
    ]
    
    try:
        # Run the command and wait for it to finish
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Look for the output video (usually in outputs/demo/ or similar)
        # Check your 'outputs' folder after a run to find the exact path
        output_path =r"D:\Videos\Gesturelsm Full Video updated.mp4" 
        
        if os.path.exists(output_path):
            return output_path
        else:
            return f"Model ran but output video not found. Error log: {result.stderr}"
            
    except Exception as e:
        return f"Error running the model: {str(e)}"

# Gradio UI
with gr.Blocks(title="Gemsy's GestureLSM App") as demo:
    gr.Markdown("# 🕺 GestureLSM: Co-Speech Gesture Generation")
    gr.Markdown("Upload an audio file and optional text to generate 3D gestures.")
    
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(type="filepath", label="Input Speech (WAV/MP3)")
            text_in = gr.Textbox(label="Transcript (Optional)", placeholder="What is being said?")
            submit_btn = gr.Button("Generate Gesture Video", variant="primary")
        
        with gr.Column():
            video_out = gr.Video(label="Generated 3D Animation")

    submit_btn.click(
        fn=generate_gesture,
        inputs=[audio_in, text_in],
        outputs=video_out
    )

if __name__ == "__main__":
    # share=True creates a public link you can send to others
    demo.launch(share=True)