"""
Functions which interact with the BreweryDB API 
"""
import json
import requests
import pandas as pd

def make_df(arr, col_header):
	df = pd.DataFrame(arr, columns=col_header)
	df[['ABV', 'IBU', 'SRM']] = df[['ABV', 'IBU', 'SRM']].apply(pd.to_numeric, errors='ignore')
	return df

def request_data(request_url, request_params):
	resp = requests.get(url=request_url, params=request_params)
	return json.loads(resp.text)

def plot_styles(df, number=10):
	pd.value_counts(df['Style']).head(n=number).iloc[::-1].plot(kind='barh')

def count_pages(request_url, request_params):
	data = request_data(request_url, request_params)
	return list(range(data['numberOfPages']))

def download_data(request_url, request_params, count, rand=False):
	header = ['Brewery', 'Beer Name', 'Style', 'ABV', 'IBU', 'SRM']
	rows = []
	if rand==True:
		request_params['order'] = 'random'
	pages = count_pages(request_url, request_params)
	for page in pages:
		request_params['p'] = page
		paginated_data = request_data(request_url, request_params)
		entries = paginated_data['data']
		for entry in entries:
			row = []
			try:
				print("\rDownloading ({} of {})".format(len(rows)+1, count), end="")
				row.extend([entry['breweries'][0]['name'],
					entry['nameDisplay'],
					entry['style']['name'],
					entry['abv'],
					entry['ibu'],
					entry['srmId']])
				if row not in rows:
					rows.append(row)
			except Exception:
				pass
			if len(rows) == count:
				df = make_df(rows, header)
				return df
				break
	df = make_df(rows, header)
	return df