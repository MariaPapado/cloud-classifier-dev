import requests

# url of the model in production
# test_url = 'https://cloud-classifier-ml.orbitaleye.nl/predict-proba'
# url of the model in the test server, replace server IP address with your server!!
test_url = 'http://0.0.0.0:8000/predict-proba'
# example image which should be GOOD but is classified as BAD
#file_path = '/img/20250511/SkySat_SKYWATCH_20250511_085520_ssc13_u0002_pansharpened_clip_0_5526330_dLgDPozUhU.tif'
file_path = '/img/20250511/SkySat_SKYWATCH_20250511_085520_ssc13_u0002_pansharpened_clip_0_5526330_dLgDPozUhU.tif'
# /mapserver/data/
response = requests.get(test_url, json={'file_path':file_path, 'url': None, 'band_type': 'BGRN'})

print(response, response.text) 
