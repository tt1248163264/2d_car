import numpy as np
from PIL import Image

model_index = 2
size = 30
model_name = ['car','obs','goal']

if model_index == 0:
    car_array = np.empty((size,size,4),dtype=np.uint8)
    mask_array = np.empty((size, size, 4), dtype=np.uint8)
    offset = size / 2 - 0.5
    th = size / 2
    for i in range(size):
        for j in range(size):
            if (i - offset)**2 + (j - offset)**2 <= (th)**2:
                car_array[i,j] = [100,100,100,255]
                mask_array[i,j] = [255,255,255,255]
            else:
                car_array[i,j] = [0,0,0,0]
                mask_array[i,j] = [0,0,0,0]

    car = Image.fromarray(car_array)
    mask = Image.fromarray(mask_array)
    car.save('car_'+str(size)+'_circle.png')
    mask.save('mask_'+str(size)+'_circle.png')
elif model_index == 1:
    obs_array = np.empty((size, size, 4), dtype=np.uint8)
    offset = size / 2 - 0.5
    th = size / 2
    for i in range(size):
        for j in range(size):
            if (i - offset)**2 + (j - offset)**2 <= (th)**2:
                obs_array[i,j] = [0,0,0,255]
            else:
                obs_array[i,j] = [0,0,0,0]

    obs = Image.fromarray(mask_array)
    obs.save('obs_'+str(size)+ '_circle.png')
#goal
elif model_index == 2:
    r = 0
    g = 200
    b = 200
    mid = size // 2
    goal_array = np.zeros((size, size, 4), dtype=np.uint8)
    goal_mask_array = np.zeros((size, size, 4), dtype=np.uint8)
    for i in range(size):
        goal_array[mid-3:mid+3,:] = [r,g,b,255]
        goal_array[:,mid-3:mid+3] = [r,g,b,255]

        goal_mask_array[mid-3:mid+3,:] = [255,255,255,255]
        goal_mask_array[:,mid-3:mid+3] = [255,255,255,255]

    goal = Image.fromarray(goal_array)
    goal_mask = Image.fromarray(goal_mask_array)
    goal.save('goal_{0}_{1}_{2}_{3}.png'.format(size,r,g,b))
    #goal_mask.save('goal_mask_{0}.png'.format(size))