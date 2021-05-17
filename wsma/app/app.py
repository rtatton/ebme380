import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly import express as px

import synthetic

STYLE = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
SUCCESS = None
UPDATE = 1
CONNECT = 2
REMOVE = 3
ERROR = dash.no_update
app = dash.Dash(__name__, external_stylesheets=STYLE)
# TODO(rdt17) Refactor to not be globals
p2d = {}
d2p = {}
last_btn = None


def dropdown_opts(dropdown):
	return [{'label': f'{k} ({v})', 'value': k} for k, v in dropdown.items()]


def get_triggered(context):
	return context.triggered[0]['prop_id'].split('.')[0]


def heading_panel():
	return html.Div([html.H2('Wearable Stress Measurement System')])


def connect_panel():
	return html.Div([
		dcc.Dropdown(
			id='dropdown',
			options=dropdown_opts(p2d),
			value='',
			placeholder='Patient ID',
			searchable=True),
		html.Div([html.Button('Connect', id='btn_connect', n_clicks=0)]),
		html.P(id='btn_connect_out', hidden=True),
		html.P(id='btn_connect_err', style={'color': 'red'})])


def save_panel():
	return html.Div([
		html.Div([
			dcc.Input(
				id='input_patient',
				value='',
				type='text',
				minLength=1,
				maxLength=128,
				placeholder='Patient ID',
				style={'width': '50%'}),
			dcc.Input(
				id='input_device',
				value='',
				type='text',
				minLength=1,
				maxLength=128,
				placeholder='Device ID',
				style={'width': '50%'})]),
		html.Button('Update', id='btn_update', n_clicks=0),
		html.Button('Remove', id='btn_remove', n_clicks=0),
		html.P(id='btn_update_out', hidden=True),
		html.P(id='btn_update_err', style={'color': 'red'}),
		html.P(id='btn_remove_out', hidden=True),
		html.P(id='btn_remove_err', style={'color': 'red'})])


def make_plots(*dfs: pd.DataFrame):
	figs = []
	for df in dfs:
		x, y = synthetic.sinusoid()
		x_label, title = df.columns
		signal = pd.DataFrame({x_label: x, title: y})
		figs.append(px.line(
			signal,
			x=x_label,
			y=title,
			title=title,
			labels={x_label: x_label, title: ''}))
	return figs


def plots():
	x, hr = synthetic.sinusoid()
	hr = pd.DataFrame({'Time (sec)': x, 'Heart Rate (bpm)': hr})
	x, edr = synthetic.sinusoid()
	edr = pd.DataFrame({'Time (sec)': x, 'Electrodermal Response (uS)': edr})
	hr, edr = make_plots(hr, edr)
	return html.Div([
		dcc.Graph(id='graph_hr', figure=hr, responsive=True, animate=True),
		dcc.Graph(id='graph_edr', figure=edr, responsive=True, animate=True)])


@app.callback(
	Output('btn_update_out', 'children'),
	Output('btn_update_err', 'children'),
	Output('btn_connect_out', 'children'),
	Output('btn_connect_err', 'children'),
	Output('btn_remove_out', 'children'),
	Output('btn_remove_err', 'children'),
	Output('dropdown', 'options'),
	Output('dropdown', 'value'),
	Input('btn_update', 'n_clicks'),
	Input('btn_connect', 'n_clicks'),
	Input('btn_remove', 'n_clicks'),
	State('input_patient', 'value'),
	State('input_device', 'value'),
	State('dropdown', 'value'))
def update_menu(s_btn, c_btn, r_btn, patient, device, dropdown):
	global last_btn
	result = []
	if not (btn := get_triggered(dash.callback_context)):
		raise PreventUpdate
	elif btn == 'btn_update':
		last_btn = UPDATE
		if not all((patient, device)):
			result.extend((ERROR, 'Provide both patient ID and device ID'))
		elif device in d2p and (owner := d2p[device]) != patient:
			result.extend((ERROR, f'Device already associated with {owner}'))
		else:
			result.extend((None, None))
			dropdown = patient
			d2p[device], p2d[patient] = patient, device
		result.extend((None, None, None, None))
	elif btn == 'btn_connect':
		last_btn = CONNECT
		result.extend((None, None))
		if not dropdown:
			result.extend((ERROR, 'Provide a patient ID'))
		elif dropdown not in p2d:
			result.extend((ERROR, 'Patient does not exist'))
		else:
			result.extend((None, None))
		result.extend((None, None))
	else:
		last_btn = REMOVE
		result.extend((None, None, None, None))
		if not dropdown:
			result.extend((ERROR, 'Provide a patient ID'))
		elif dropdown not in p2d:
			result.extend((ERROR, 'Patient does not exist'))
		else:
			result.extend((None, None))
			d2p.pop(p2d.pop(dropdown))
			dropdown = None
	result.extend((dropdown_opts(p2d), dropdown))
	return result


@app.callback(
	Output('graph_hr', 'figure'),
	Output('graph_edr', 'figure'),
	Input('btn_connect_out', 'children'),
	Input('btn_connect_err', 'children'))
def update_plots(out, err):
	if any((out, err, last_btn != CONNECT)):
		raise PreventUpdate
	if last_btn == CONNECT:
		x, hr = synthetic.sinusoid()
		hr = pd.DataFrame({'Time (sec)': x, 'Heart Rate (bpm)': hr})
		x, edr = synthetic.sinusoid()
		edr = {'Time (sec)': x, 'Electrodermal Response (uS)': edr}
		edr = pd.DataFrame(edr)
		return make_plots(hr, edr)


def settings():
	return html.Div([
		heading_panel(),
		save_panel(),
		html.Br(),
		connect_panel()])


def layout():
	return html.Div([settings(), plots()])


app.layout = layout()

if __name__ == '__main__':
	app.run_server(debug=False)
