import os
import replicate
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
import requests
import json

# Load environment variables
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_USER_NAME = os.getenv("REPLICATE_USER_NAME")
# Initialize Replicate client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify your frontend URL here, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class TrainRequest(BaseModel):
    steps: int = 1000
    lora_rank: int = 16
    optimizer: str = "adamw8bit"
    batch_size: int = 1
    resolution: str = "512,768,1024"
    autocaption: bool = True
    trigger_word: str = "TOK"
    learning_rate: float = 0.0004
    train_text_encoder: bool = True
    checkpointing_steps: int = 500
    max_train_steps: int = 2000
    prior_preservation: bool = False
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    mixed_precision: str = "fp16"

# Function to upload a ZIP file to Replicate
def upload_zip_to_replicate(zip_file: UploadFile):
    files = {"content": (zip_file.filename, zip_file.file, "application/zip")}
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}
    
    response = requests.post("https://api.replicate.com/v1/files", headers=headers, files=files)

    if response.status_code == 201:
        return response.json()["urls"]["get"]  # Get file URL
    else:
        raise HTTPException(status_code=500, detail=f"File upload failed: {response.text}")

@app.post("/train-model/")
async def train_model(zip_file: UploadFile = File(...), payload: str = Form(...), modelname: str = Form(...)):
    try:
        # Upload the ZIP file first
        zip_url = upload_zip_to_replicate(zip_file)

        # print(type(json.loads(json.loads(payload))))

        # Add the uploaded file URL to the training payload
        training_data = json.loads(json.loads(payload))
        training_data["input_images"] = zip_url

        destination = f"{REPLICATE_USER_NAME}/{modelname}"

        if not destination:  # If destination is empty or invalid, raise an error
            raise HTTPException(status_code=400, detail="Destination cannot be empty")

        # Start training
        training = replicate_client.trainings.create(
            destination=destination,
            version="ostris/flux-dev-lora-trainer:b6af14222e6bd9be257cbc1ea4afda3cd0503e1133083b9d1de0364d8568e6ef",
            input=training_data,
        )

        return {
            "message": "Training started successfully",
            "training_id": training.id,
            "status_url": f"https://replicate.com/trainings/{training.id}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training-status/{training_id}")
async def get_training_status(training_id: str):
    try:
        # Fetch training status from Replicate
        training = replicate_client.trainings.get(training_id)

        return {
            "training_id": training.id,
            "status": training.status,  # Can be "starting", "processing", "succeeded", "failed"
            "created_at": training.created_at,
            "completed_at": training.completed_at,
            # "output_model": training.output_model if training.status == "succeeded" else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Welcome to Replicate Model Manager API"}

# @app.get("/list-models/")
# def list_models():
#     """
#     Fetch all custom models created on your Replicate account.
    
#     Returns:
#     - List of models with name, owner, visibility, and URL.
#     """
#     try:
#         # Fetch all models
#         models = list(replicate_client.models.list())

#         # Format response
#         model_list = [
#             {
#                 "model_name": model.name,
#                 "owner": model.owner,
#                 "visibility": model.visibility,
#                 "url": f"https://replicate.com/{model.owner}/{model.name}"
#             }
#             for model in models
#         ]

#         return {"models": model_list}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-model/")
async def create_custom_model(
    model_name: str = Form(...),
    hardware: str = Form(...),
    visibility: str = Form(...),
    owner: str = Form(...),  # Adding owner as a parameter
):
    try:
        # Prepare the model creation parameters
        model_data = {
            "name": model_name,
            "hardware": hardware,
            "visibility": visibility,
            "owner": owner  # Owner is required by Replicate API
        }

        # Create the model using replicate client
        model = replicate_client.models.create(**model_data)

        # Return success message with model details
        return {
            "message": "Model created successfully",
            "model_id": model.id,
            "model_url": f"https://replicate.com/{owner}/{model_name}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def generate_prediction(
    version: str = Form(...),  # Version is passed as input from form data
    prompt: str = Form(...),
    my_model: str = Form(...),
    model: str = Form(...),  # Model is passed as input from form data
    go_fast: bool = Form(False),
    lora_scale: float = Form(1),
    megapixels: str = Form("1"),
    num_outputs: int = Form(1),
    aspect_ratio: str = Form("1:1"),
    output_format: str = Form("webp"),
    guidance_scale: float = Form(3),
    output_quality: int = Form(80),
    prompt_strength: float = Form(0.8),
    extra_lora_scale: float = Form(1),
    num_inference_steps: int = Form(28),
    height: int = Form(512),  # Added height parameter
    width: int = Form(512),   # Added width parameter
    seed: int = Form(42),     # Added seed for randomization control
    temperature: float = Form(1.0),  # Added temperature for controlling randomness
    top_p: float = Form(1.0),  # Added top_p for controlling diversity
):
    try:
        # Prepare the input for prediction
        input_data = {
            "prompt": prompt,
            "model" : model,
            "go_fast": go_fast,
            "lora_scale": lora_scale,
            "megapixels": megapixels,
            "num_outputs": num_outputs,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "guidance_scale": guidance_scale,
            "output_quality": output_quality,
            "prompt_strength": prompt_strength,
            "extra_lora_scale": extra_lora_scale,
            "num_inference_steps": num_inference_steps,
            "height": height,  # Added height
            "width": width,    # Added width
            "seed": seed,      # Added seed
            "temperature": temperature,  # Added temperature
            "top_p": top_p,    # Added top_p
        }

        # Ensure model and version are properly formatted
        model_name_with_version = f"{REPLICATE_USER_NAME}/{my_model}:{version}"

        print( model_name_with_version)

        # Run the prediction using replicate.run() with the model and version
        output = replicate_client.run(
            model_name_with_version,
            input=input_data
        )

        for text in output:
            print(text, end="")

        # result = ''.join(output)
        result = ''.join(str(text) for text in output)
        

        # Return the image URL or the result object directly
        return {
            "message": "Prediction generated successfully",
            "prediction": result  # Return the output directly here
        }

    except Exception as e:
        # Log the exception message for debugging
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/upscale/")
async def upscale_image(
    image_url: str = Form(...),  
    scale: int = Form(2),  
    denoise: float = Form(0.5),  
    face_enhance: bool = Form(False)  
):
    try:
        # Hardcoded model version and name
        model_name_with_version = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"

        # Prepare the input for upscaling
        input_data = {
            "image": image_url,
            "scale": scale,
            "denoise": denoise,
            "face_enhance": face_enhance
        }

        # Run the upscale process
        output = replicate_client.run(
            model_name_with_version,
            input=input_data
        )

        result = ''.join(str(text) for text in output)

        # for index, item in enumerate(output):
        # with open(f"output_{index}.png", "wb") as file:
        # file.write(item.read())

        # Return the upscaled image URL(s)
        return {
            "message": "Image upscaled successfully",
            "upscaled_images": result  # This will return a list of image URLs
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")