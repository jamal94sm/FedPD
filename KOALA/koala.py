import numpy as np
import torch
from . import myPlayers
from . import myModels
from . import myUtils




def run(distributed_dataset, num_classes, name_classes, synthetic_public_data, original_public_data, args):

    FM, processor, tokenizer = myModels.load_clip_model(args)
    server_model = myModels.Image_prompting_plus_Fm(FM, processor, tokenizer, num_classes, name_classes, args).to(args.device)

    clients = [ myPlayers.Device( id, distributed_dataset[id], num_classes, name_classes, synthetic_public_data, args ) for id in range(args.num_clients) ]
    clients = clients[:min(len(clients), 3)] #only three of them is enough

    server = myPlayers.Server(clients, server_model, synthetic_public_data, args)



    for round in range(args.rounds):
        print("=" * 20, f" Round {round + 1}/{args.rounds} ", "=" * 20)

        for client in clients:
            client.local_training(client.data)
            print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
            if round > 0 :  
                client.local_distillation(
                    original_public_data,
                    general_knowledge, 
                    proto = True if "proto" in args.setup else False,
                    )

        global_model = server.fedavg_aggregation()
        train_loader = torch.utils.data.DataLoader(original_public_data['train'], batch_size=32)
        global_model.eval()
        global_logits = torch.cat([ global_model(batch['image'].to(args.device)).cpu() for batch in torch.utils.data.DataLoader(original_public_data['train'], batch_size=32) ], dim=0)
        server.distill_generator(original_public_data, global_logits)
        general_knowledge = server.get_general_knowledge()   

    avg_test_Acc = np.mean([client.test_Acc for client in clients], axis=0)
    return avg_test_Acc



