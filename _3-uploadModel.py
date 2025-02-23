from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import upload_file
from huggingface_hub.utils import HfHubHTTPError

api = HfApi()
repo_name = "autopilot-car-model"  # Change to your preferred name
username = "Seb0099"  # Change to your Hugging Face username
repo_id = f"{username}/{repo_name}"

try:
    api.create_repo(repo_id=f"{username}/{repo_name}", private=False)
except HfHubHTTPError as e:
    if "already exists" in str(e):
        print(f"Repository '{repo_id}' already exists.")
    else:
        print(f"Error: {e}")

# Define model path
model_path = "image_classifier.keras"

# Upload model
upload_file(
    path_or_fileobj=model_path,
    path_in_repo="image_classifier.keras",  # Name inside repo
    repo_id=f"{username}/{repo_name}",
)
print("Model uploaded successfully!")

upload_file(
    path_or_fileobj="model.tflite",
    path_in_repo="model.tflite",
    repo_id=f"{username}/{repo_name}",
)
print("TFLite model uploaded successfully!")
