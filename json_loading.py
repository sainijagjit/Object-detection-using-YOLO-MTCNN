import pandas as pd
import numpy as np

import json
with open('via_project_23Nov2019_22h18m44s.json') as json_file:
    data = json.load(json_file)
data=json.dumps(data,indent=4,sort_keys=True)
print(data)
for p in data['people']:
    print('Name: ' + p['name'])
    print('Website: ' + p['website'])
    print('From: ' + p['from'])
    print('')