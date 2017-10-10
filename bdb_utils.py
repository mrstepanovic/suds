"""
PUT DOCS HERE
"""

def request_data(request_url, request_params):
    resp = requests.get(url=request_url, params=request_params)
    return json.loads(resp.text)

def download_beer_data(request_url, request_params, count):
    header = ['Brewery', 'Beer Name', 'Style', 'ABV', 'IBU', 'SRM', 'OG']
    beers = []
    data = request_data(request_url, request_params)
    for page in range(data['numberOfPages']):
        request_params['p'] = page
        fresh_data = request_data(request_url, request_params)
        for i in fresh_data['data']:
            beer = []
            try:
                print("\rDownloading ({} of {})".format(len(beers), count), end="")
                beer.extend([i['breweries'][0]['name'],
                            i['nameDisplay'],
                            i['style']['name'],
                            i['abv'],
                            i['ibu'],
                            i['srmId'],
                            i['originalGravity']])
                beers.append(beer)
            except Exception:
                pass
            if len(beers) >= count:
                return pd.DataFrame(beers, columns=header)
                break
    return pd.DataFrame(beers, columns=header) 