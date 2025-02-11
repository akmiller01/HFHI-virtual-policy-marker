import os
import requests
from dotenv import load_dotenv

def load_api_key():
    load_dotenv()
    return os.getenv("LAMBDA_LABS_KEY")

def get_running_instances(api_key):
    url = "https://cloud.lambdalabs.com/api/v1/instances"
    response = requests.get(url, auth=(api_key, ""), headers={"Content-Type": "application/json"})
    
    if response.status_code != 200:
        print(f"Error fetching instances: {response.text}")
        return []
    
    data = response.json().get("data", [])
    return [instance["id"] for instance in data]

def terminate_instances(api_key, instance_ids):
    url = "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate"
    payload = {"instance_ids": instance_ids}
    response = requests.post(url, auth=(api_key, ""), json=payload, headers={"Content-Type": "application/json"})
    
    if response.status_code == 200:
        print("Successfully terminated instances:", instance_ids)
    else:
        print(f"Error terminating instances: {response.text}")

def main():
    api_key = load_api_key()
    if not api_key:
        print("Error: LAMBDA_LABS_KEY not found in .env file")
        return
    
    instance_ids = get_running_instances(api_key)
    if not instance_ids:
        print("No running instances found.")
        return
    
    terminate_instances(api_key, instance_ids)

if __name__ == "__main__":
    main()
