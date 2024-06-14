from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def main():
    from dotenv import load_dotenv
    try:
        load_dotenv()
        prediction_endpoint=os.getenv('PredictionEndpoint')
        prediction_key=os.getenv('PredictionKey')
        project_id=os.getenv('ProjectID')
        model_name=os.getenv('ModelName')
        
        # Authenticate the client
        credential = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credential)
        
        # Load the image
        image_file = 'image11.jpeg'
        print('Detecting objects in', image_file)
        image = Image.open(image_file)
        h, w, ch = np.array(image).shape
        
        with open(image_file, mode="rb") as image_data:
            results = prediction_client.detect_image(project_id, model_name, image_data)
            
        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')
        
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        lineWidth = int(w / 100)
        color = 'magenta'
        
        for prediction in results.predictions:
            if (prediction.probability * 100) > 50:
                left = prediction.bounding_box.left * w
                top = prediction.bounding_box.top * h
                height = prediction.bounding_box.height * h
                wdth = prediction.bounding_box.width * w
                
                points = [(left, top), (left + wdth, top), (left + wdth, top + height), (left, top + height), (left, top)]
                draw.line(points, fill=color, width=lineWidth)
                plt.annotate(f"{prediction.tag_name}: {prediction.probability:.2f}%", (left, top), color='white', fontsize=12, backgroundcolor='magenta')
        
        plt.imshow(image)
        outputfile = 'output.jpg'
        fig.savefig(outputfile, bbox_inches='tight')
        print('Results saved in', outputfile)
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    main()
