import uvicorn
from fastapi import FastAPI, File, UploadFile

import prediction as pd

app = FastAPI()

@app.get('/index')
def hello_world(name: str):
    return f"Hello {name}!"

@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png","gif")
    if not extension:
        return "Image must be jpg,png and gif format!"
    image = pd.read_image(await file.read())
    image = pd.preprocess(image)

    pred = pd.predict(image)
    print(pred)
    return pred


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
