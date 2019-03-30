from test_gen import train_network



GLR = [0.00001,0.0001,0.0002,0.0005,0.001,0.01,0.1]
DLR = [0.00001,0.0001,0.0002,0.0005,0.001,0.01,0.1]

bias = 0.5 
for glr in GLR:
    for dlr in DLR:
        for runNo in range(1,2):
            train_network(glr,dlr,runNo,bias)