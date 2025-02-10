#!/bin/bash
python create_rs_map.py --data_type NYT-Exact
python create_rs_map.py --data_type NYT-Partial
python create_rs_map.py --data_type WebNLG-Exact
python create_rs_map.py --data_type WebNLG-Partial

python prepro.py --data_type NYT-Exact
python prepro.py --data_type NYT-Partial
python prepro.py --data_type WebNLG-Exact
python prepro.py --data_type WebNLG-Partial
