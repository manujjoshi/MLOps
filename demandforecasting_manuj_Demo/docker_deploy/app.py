from fastapi import FastAPI
import uvicorn
import prophet
import joblib


app = FastAPI()

@app.post("/process_inputs")
async def process_inputs(store_id: str, item_id: str, period: int):
    model = f'model_{store_id}_{item_id}'
    print(model)
    filename =rf'outputs/{model}'
    loaded_model = joblib.load(open(filename, 'rb'))
    df = loaded_model.make_future_dataframe(periods=period,freq='d',include_history=True)
    print("-"*100)
    df = df.tail(period)
    print("df", df)
    pred = loaded_model.predict(df)
    dd = pred[["ds","yhat"]]
    return {'predictions': dd}


if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8080)
