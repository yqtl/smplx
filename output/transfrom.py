import pickle

import io

import numpy as np

w=np.load(io.BytesIO(open('case2_male_custom_shape.pkl', 'rb').read()),allow_pickle=True)

pickle.dump(w, open("case2_male_custom_shape_2.pkl","wb"), protocol=2)