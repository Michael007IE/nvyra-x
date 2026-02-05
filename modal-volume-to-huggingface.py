# ------------------------------------------------------------------------------
#  CONFIDENTIAL AND PROPRIETARY
#  Copyright (c) 2025-2026 nvyra-x. All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of
#  nvyra-x. The intellectual and technical concepts contained herein are
#  proprietary to nvyra-x and may be covered by U.S. and Foreign Patents,
#  patents in process, and are protected by trade secret or copyright law.
#  Dissemination of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained from nvyra-x.
# ------------------------------------------------------------------------------

%%writefile upload.py
import modal
import os


vol = modal.Volume.from_name("reasoning-data")
secret = modal.Secret.from_name("huggingface-secret")
image = modal.Image.debian_slim().pip_install("huggingface_hub")
app = modal.App("volume-exporter")
@app.function(
    image=image,
    volumes={"/data": vol}, 
    secrets=[secret],       
    timeout=3600            
)
def upload_to_huggingface(repo_id, repo_type="dataset"):
    from huggingface_hub import HfApi
    print(f"Preparing to upload '/data' to {repo_id}...")
    api = HfApi()
    print(f"Ensuring repository {repo_id} exists...")
    api.create_repo(
        repo_id=repo_id, 
        repo_type=repo_type, 
        exist_ok=True,    
        private=True     
    )
    print("Starting upload...")
    api.upload_folder(
        folder_path="/data",
        repo_id=repo_id,
        repo_type=repo_type, 
        path_in_repo=".",
    )
    print("Upload complete!")
@app.local_entrypoint()
def main():
    """Upload the reasoning data volume to Hugging Face."""
    upload_to_huggingface.remote(repo_id="Feargal/reasoning-data", repo_type="dataset")

# !modal run --detach upload.py
