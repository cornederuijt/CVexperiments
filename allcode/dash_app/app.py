import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = app.layout = html.Div(children=[
    html.H1(children='Is it a cat, or is it a dog?',
            style={
                'textAlign': 'center'
            }),

    # Not sure how to center the button...
    dcc.Upload(html.Button('Upload File',
                           id='upload_button',
                           style={
                               'margin': '0',
                               'width': '144px'
                           }),
               id="upload_image"),
    html.Div([
        html.Img(id='uploaded_image', width=300),
    ], className="five columns")
])

@app.callback(Output('uploaded_image', 'src'),
              [Input('upload_image', 'contents')])
def update_output(image):
    return image


if __name__ == '__main__':
    app.run_server(debug=True)