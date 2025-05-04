from flask import Flask, render_template, request, redirect, url_for
import os
from predict import predict_tumor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Predict tumor
        result = predict_tumor(file_path)
        
        # Map result to "Yes" or "No"
        result_text = "Yes" if result == "Tumor" else "No"
        
        return render_template('index.html', result=result_text, image_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

