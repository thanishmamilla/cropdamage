# from flask import Flask,render_template,request,jsonify,send_file
# from flask_cors import CORS
# import os
# from surys import CDNetPredictor


# predictor = CDNetPredictor()
# class Args:
#     pass



# app=Flask(__name__)

# CORS(app)

# @app.route('/',methods=["POST","GET"])
# def home():
#     return "server successful"

# @app.route('/mura_image',methods=['POST'])
# def staellite():
#     # if(request.method=="POST"):
#     #     a=request.form['image1']
#     #     b=request.form['image2']
#     bimage = request.files['bimage']
#     aimage = request.files['aimage']

#     # Define upload directory
#     upload_dir = 'repeatfolder'
#     if not os.path.exists(upload_dir):
#         os.makedirs(upload_dir)

#     # Save the uploaded images
#     bimage_path = os.path.join(upload_dir,"time1", "image1.png")
#     aimage_path = os.path.join(upload_dir,"time2","image1.png" )
#     bimage.save(bimage_path)
#     aimage.save(aimage_path)
#     # Define your options
#     opt = Args()
#     opt.gpu_id = "0"  # Specify GPU ID if multiple GPUs are available
#     opt.model_dir = 'netCD_epoch_43.pth'  # Path to the trained model
#     opt.batch_size = 8  # Batch size for prediction
#     opt.crop_size = 512  # Crop size for input images
#     opt.path_img1 = 'repeatfolder/time1'  # Path to directory containing time 1 images
#     opt.path_img2 = 'repeatfolder/time2'  # Path to directory containing time 2 images
#     opt.save_dir = 'Output_images1/'  # Directory to save predicted images
#     opt.suffix = '.png'  # Suffix of image files
#     predictor.predict_images(opt)
#     image_path="Output_images1/result1.png"

#     # Return a response
#     return send_file(image_path, mimetype='image/jpeg')



# if( __name__ == '__main__'):
#     app.run(port=5000,debug=True)




from flask import Flask, request, send_file
from flask_cors import CORS
import os
from surys import CDNetPredictor
from PIL import Image

app = Flask(__name__)
CORS(app)



class Args:
    pass

@app.route('/', methods=["POST", "GET"])
def home():
    return "Server is running successfully"

@app.route('/mura_image', methods=['POST'])
def staellite():
    bimage = request.files['bimage']
    aimage = request.files['aimage']

    # Define upload directory
    predictor = CDNetPredictor()
    upload_dir = 'repeatfolder'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the uploaded images
    bimage_path = os.path.join(upload_dir, "time1", "image1.png")
    aimage_path = os.path.join(upload_dir, "time2", "image1.png")
    bimage.save(bimage_path)
    aimage.save(aimage_path)

    # Define your options
    opt = Args()
    opt.gpu_id = "0"  # Specify GPU ID if multiple GPUs are available
    opt.model_dir = 'netCD_epoch_43.pth'  # Path to the trained model
    opt.batch_size = 8  # Batch size for prediction
    opt.crop_size = 512  # Crop size for input images
    opt.path_img1 = 'repeatfolder/time1'  # Path to directory containing time 1 images
    opt.path_img2 = 'repeatfolder/time2'  # Path to directory containing time 2 images
    opt.save_dir = 'Output_images1/'  # Directory to save predicted images
    opt.suffix = '.png'  # Suffix of image files

    # Predict images using the predictor
    predictor.predict_images(opt)

    # Define the path of the predicted image
    image_path = "Output_images1/result1.png"

    # Return the predicted image as a response
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
