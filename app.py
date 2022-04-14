from multiprocessing.sharedctypes import Value
from flask import Flask, render_template, request
from collections import Counter
from sklearn.cluster import KMeans
import cv2


app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('final.html')


@app.route('/image', methods=['GET'])
def home():
    return render_template('finalimage.html')


@app.route('/imgtopdf')
def ipdf():
    return render_template('imagetopdf.html')


@app.route('/image', methods=['POST'])
def upload():
    imagefile = request.files['fimg']
    image_path = "./static/upload/"+imagefile.filename
    # num = request.form.get("cnum", type=int)
    # print(num)
    imagefile.save(image_path)

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(image, (1200, 600))

    def RGB2HEX(color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    modified_image = cv2.resize(
        image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(
        modified_image.shape[0]*modified_image.shape[1], 3)

    clf = KMeans(n_clusters=8)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    return render_template('finalimage.html', upimage=hex_colors)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
