from fastapi import FastAPI, UploadFile, File, HTTPException,Request,Depends
from fastapi.responses import JSONResponse, FileResponse
from moviepy.editor import VideoFileClip
import whisper
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from typing import List
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import uvicorn

load_dotenv()

API_ACCESS_KEY = os.getenv("API_ACCESS_KEY")


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],            # or ["https://their‑frontend.com"]
  allow_methods=["*"],
  allow_headers=["*"],
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
os.environ["PATH"] += os.pathsep + r"E:\Video-To-Action-Points\ffmpeg\bin"


def verify_api_key(request: Request):
    client_key = request.headers.get("x-api-key")  # You can also use query params
    if not client_key or client_key != API_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")
    return True

# def load_text_file(file_path: str) -> List[Document]:
#     try:
#         loader = TextLoader(file_path, encoding="utf-8")
#         return loader.load()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to load text file: {e}")


# def load_document(file_path: str) -> List[Document]:
#     if file_path.endswith(".txt"):
#         return load_text_file(file_path)
#     raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_path}")


def generate_action_points(text:str) -> str:
    try:
        response = llm.invoke(
            f"""You have this text {text}.Give me the summary of the text."""
        )
        return response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM summary generation failed: {e}")


# def save_action_points_pdf(action_points: str, file_name: str):
#     try:
#         pdf = FPDF()
#         pdf.set_auto_page_break(auto=True, margin=8)
#         pdf.add_page()
#         pdf.set_font("Arial", size=12)
#         pdf.multi_cell(0, 8, action_points)
#         pdf_path = f"{file_name}_action_points.pdf"
#         pdf.output(pdf_path)
#         return pdf_path
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save PDF: {e}")


def extract_audio_from_video(video_path: str, audio_output_path: str):
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_output_path)
        audio_clip.close()
        video_clip.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e}")


def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {e}")


# def save_transcript(transcript: str, output_file: str):
#     try:
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(transcript)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save transcript: {e}")


@app.post("/video_to_text")
async def video_to_text(video: UploadFile = File(...), model_size: str = "base",authorized: bool = Depends(verify_api_key)):
    try:
        file_name = video.filename
        # name = os.path.splitext(file_name)[0]
        # temp_video_path = f"temp_{file_name}"

        # Save uploaded file locally

        # if not file_name.endswith(".mp4"):
        #     os.remove(temp_video_path)
        #     raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .mp4 video file.")

        intermediate_audio_path = "extracted_audio.wav"
        # output_text_path = f"{name}_transcription.txt"

        # Process video → audio → text
        extract_audio_from_video(file_name, intermediate_audio_path)
        transcript = transcribe_audio(intermediate_audio_path, model_size)
        # save_transcript(transcript, output_text_path)

        # Generate action points
        # text_docs = load_document(transcript)
        action_points = generate_action_points(transcript)

        # Save action points to PDF
        # pdf_path = save_action_points_pdf(action_points, name)

        # Cleanup
        # os.remove(temp_video_path)
        if os.path.exists(intermediate_audio_path):
            os.remove(intermediate_audio_path)

        return JSONResponse(
            content={
                "status": "success",
                "summary": action_points,
                "transcript": transcript,
            }
        )

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5500,reload=True)
