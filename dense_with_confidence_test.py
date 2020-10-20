import numpy as np

from dense_with_confidence import ConfidenceMLP

f = ConfidenceMLP(3,[50,50])
data_size = 100000
data1 = np.random.rand(data_size,1)
data2 = np.random.rand(data_size,1)
data3 = np.random.rand(data_size,1)
noise1 = np.random.rand(data_size,1)
noise2 = np.random.rand(data_size,1)
noise3 = np.random.rand(data_size,1)

data_x = np.concatenate([data1, data2, data3], axis=1)
data_y = np.concatenate([data1, data2, noise3], axis=1)

def show_predictions(n, obs):
    print('=========================================')
    print(f' observation = {obs}')
    for _ in range(n):
        y = f.predict(obs)
        print(f' predicton = {y[0]} | conf = {y[1]}')
    print('*****************************************')

show_predictions(1, [0,0,0])
show_predictions(1, [0.5,0.5,0])

f.fit(data_x, data_y, 5)

print('After training')


show_predictions(1, [0,0,0])
show_predictions(1, [0.3,0.2,0.7])

