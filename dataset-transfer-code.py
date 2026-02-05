# ------------------------------------------------------------------------------
#  CONFIDENTIAL AND PROPRIETARY
#  Copyright (c) 2025-2026 nvyra-x. All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of
#  nvyra-x. The intellectual and technical concepts contained herein are
#  proprietary to nvyra-x and may be covered by Irish and Foreign Patents,
#  patents in process, and are protected by trade secret or copyright law.
#  Dissemination of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained from nvyra-x.
# ------------------------------------------------------------------------------
# Used for importing different volumes into files then into other volumes - e.g. sharing volumes without having to pay for a Modal Pro Subscription.
# %%writefile import_zip.py
import modal
import os
import zipfile

app = modal.App("volume-importer")
vol = modal.Volume.from_name("rag-harvest-storage-prod", create_if_missing=True)

@app.function(volumes={"/data": vol}, timeout=1200)
def upload_volume(zip_bytes):
    print(" Received zip bytes. Saving to temp...")
    with open("/tmp/incoming.zip", "wb") as f:
        f.write(zip_bytes)
        
    print(" Extracting into Volume...")
    with zipfile.ZipFile("/tmp/incoming.zip", 'r') as z:
        z.extractall("/data")
        
    # Validation
    count = len(os.listdir("/data"))
    print(f" Success! Volume now contains {count} items.")

@app.local_entrypoint()
def main():
    input_file = "transfer_data.zip"
    if not os.path.exists(input_file):
        print(f" Error: {input_file} not found.")
        return

    print(f" Uploading {input_file} to new account...")
    with open(input_file, "rb") as f:
        data = f.read()
        
    upload_volume.remote(data)
    print(" Transfer Complete!")

# !modal run --detach import_zip.py
