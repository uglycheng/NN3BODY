import numpy as np
import random

def sample_data(index):
    dataset = []
    for i in index:
        trajectory = []
        no_error = True
        data[i] = data[i].strip("\n").split(",") 
        init_pos = data[i][1:]
        data[i] = data[i][0:1] + init_pos + data[i][1:]
        data[i] = np.array(data[i])
        try:
            data[i] = data[i].astype(np.float32)
        except ValueError:
            continue
        for val in data[i]:
            if val > 10**19:
                no_error = False
                break
        if not no_error:
            continue
        trajectory.append(data[i])
        for j in range(i+1,i+101):
            data[j] = data[j].strip("\n").split(",")
            data[j] = data[j][0:1] + init_pos + data[j][1:]
            data[j] = np.array(data[j])
            try:
                data[j] = data[j].astype(np.float32)
            except ValueError:
                no_error = False
                break
            for val in data[j]:
                if val > 10**19:
                    no_error = False
                    break
            if not no_error:
                break
            trajectory.append(data[j])
        if no_error:
            dataset += trajectory
    dataset = np.array(dataset)
    return dataset

if __name__ == "__main__":
    data = open("data.csv")
    data = data.readlines()
    random.seed(29)
    test_index = random.sample(range(0,len(data),101),int(len(data)/(101*5)))
    test_index.sort()
    train_index = list(set(range(0,len(data),101)) - set(test_index))
    test = sample_data(test_index)
    train = sample_data(train_index)
    np.save("trainset_sample",train)
    np.save("testset_sample",test)


