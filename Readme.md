Store all files in the same folder.
Create one .env file and the contents : 
REPLICATE_API_TOKEN=<your replicate token>

Install python and pip.
then run pip install -r requirements.txt

then run the pyhon file having fastAPI using:
uvicorn main:app --reload

test the app is started or not by entering in browser:
127.0.0.1:8000 
message will be "Welcome to Replicate Model Manager API"

you can check the apis in the following link:
http://127.0.0.1:8000/docs

now download the Postman application(api testing tool) to test the apis.

Follow the videos for more

