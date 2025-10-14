import numpy as np
from . import myPlayers
from . import myModels
from . import myUtils



def run(distributed_dataset, num_classes, name_classes, public_data, args):

    clients = [ myPlayers.Device( id, distributed_dataset[id], num_classes, name_classes, public_data, args ) for id in range(args.num_clients) ]
    server = myPlayers.Server(clients)

    for round in range(args.rounds):
        print("=" * 20, f" Round {round + 1}/{args.rounds} ", "=" * 20)

        for client in clients:
            client.local_training(client.data)
            print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
            if round > 0 :  
                client.local_distillation(client.public_data, agg)
            client.cal_logits( 
                client.public_data,
                sifting = True if "sift" in args.setup else False,
                )
        agg = server.aggregation()


    avg_test_Acc = np.mean([client.test_Acc for client in clients], axis=0)
    return avg_test_Acc






    