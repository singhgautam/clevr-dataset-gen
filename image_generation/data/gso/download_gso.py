import os
import json

from tqdm import tqdm
from collections import defaultdict

gso_json = 'GSO.json'

category_counts = defaultdict(int)

with open(gso_json) as json_file:
    data = json.load(json_file)
    items = list(data['assets'].keys())
    for i, item in tqdm(enumerate(items)):
        category = data['assets'][item]['metadata']['category']
        category_counts[category] += 1
        if category in ['Shoe', 'Bottles and Cans and Cups', 'Consumer Goods']:
            os.system(f"wget https://storage.googleapis.com/kubric-public/assets/GSO/{item}.tar.gz")
            os.system(f"mkdir {item}")
            os.system(f"tar -xf {item}.tar.gz -C {item}")
            os.system(f"rm {item}.tar.gz")
