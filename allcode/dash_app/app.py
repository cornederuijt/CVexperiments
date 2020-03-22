import dash
import base64
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from allcode.controllers.app_controller.simple_image_manager import SimpleImageManager
from PIL import Image
from PIL.JpegImagePlugin import Image
import numpy as np
import cv2
import io

# Ths should be become a separate service, and the app's details should be stored in the app_config.py:
image_dir = "./data/cat_dog_images"
# image_manager = simple_image_manager(image_dir, )

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = app.layout = html.Div(children=[
    html.H1(children='Is it a cat, or is it a dog?',
            style={
                'textAlign': 'center'
            }),

    # Not sure how to center the button...
    dcc.Upload(html.Button('Upload File',
                           id='upload_button'),
               id="upload_image"),

    html.H3(children='Uploaded image',
            id="uploaded_image_title",
            style={'marginLeft': 700}),

    html.Div([
        html.Img(id='uploaded_image', width=300),
    ], className="five columns", style={'marginLeft': 650, "marginTop": 25}),

    # html.H3(children='5 Most comparable images',
    #         id="knn_title",
    #         style={'marginLeft': 700}),


    html.Div([
        html.Table(
            [html.Tr([
                html.Td(html.Img(id='knn_image1', width=300)),
                html.Td(html.Img(id="knn_image2", width=300)),
                html.Td(html.Img(id="knn_image3", width=300)),
                html.Td(html.Img(id="knn_image4", width=300)),
                html.Td(html.Img(id="knn_image5", width=300))])],
            id='res_table2')])

])


@app.callback([Output('uploaded_image', 'src'),
               Output('knn_image1', 'src'),
               Output('knn_image2', 'src'),
               Output('knn_image3', 'src'),
               Output('knn_image4', 'src'),
               Output('knn_image5', 'src')],
              [Input('upload_image', 'contents')])
def update_output(image):
    if image is None: #  Do not update if no image is selected
        raise dash.exceptions.PreventUpdate
    image_dir = "./data/cat_dog_images"
    k_in_knn = 5
    image_manager = SimpleImageManager(image_dir, k_in_knn)

    #TODO: Check FLASK, perhaps answers there rather than in Dash directly
    image = Image.open(io.BytesIO(base64.b64decode(image)))
    image.show()

    nparr = np.fromstring(base64.b64encode(image), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.COLOR_BGR2GRAY)  # cv2.IMREAD_COLOR in OpenCV 3.1

    res = image_manager.processes_image(img_np)

    enc_images = [base64.b64encode(open(im, "rb").read()) for im in res.knn5_image_locations['image_loc'].tolist()]
    src_ready_images = ['data:image/jpg;base64,{}'.format(im.decode()) for im in enc_images]

    return image, \
           src_ready_images[0], \
           src_ready_images[1], \
           src_ready_images[2], \
           src_ready_images[3], \
           src_ready_images[4]


if __name__ == '__main__':
    app.run_server(debug=True)