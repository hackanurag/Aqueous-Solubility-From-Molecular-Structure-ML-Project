import sys
import pandas as pd
import requests

try:
    r = requests.get('https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv')
    if(r.status_code == 200):
        sol = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv')
except Exception as e:
    print (e)