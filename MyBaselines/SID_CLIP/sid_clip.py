import numpy as np
import torch
from . import myPlayers
from . import myModels
from . import myUtils




def run(distributed_dataset, num_classes, name_classes, public_data, args):

    FM, processor, tokenizer = myModels.load_clip_model(args)
    server_model = myModels.Image_prompting_plus_Fm(FM, processor, tokenizer, num_classes, name_classes, args).to(args.device)

    clients = [ myPlayers.Device( id, distributed_dataset[id], num_classes, name_classes, public_data, args ) for id in range(args.num_clients) ]
    clients = clients[:min(len(clients), 3)] #only three of them is enough
    server = myPlayers.Server(clients, server_model, public_data, args)



    teacher_model = myModels.clip_plus_linear_head(FM, processor, tokenizer, num_classes, name_classes, args.device).to(args.device)
    few_shot_data = myUtils.get_few_shot_subset(clients[0].data, num_shots=5)
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=args.local_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    loss_fn = torch.nn.functional.cross_entropy
    myUtils.Train(teacher_model, 
                few_shot_data, 
                optimizer, 
                scheduler, 
                loss_fn,  
                batch_size = 8, 
                epochs = args.rounds,
                device = args.device,
                debug = False)   
    teacher_logits = teacher_model.inference(clients[0].public_data)
    del teacher_model
    torch.cuda.empty_cache()


    for round in range(args.rounds):
        print("=" * 20, f" Round {round + 1}/{args.rounds} ", "=" * 20)
    
        for client in clients:
            client.local_distillation(
                client.public_data,
                teacher_logits
                )
            client.test_Acc.append( myUtils.Evaluate(client.model,  client.data["test"]["image"], client.data["test"]["label"], args.device)[0] )
            print(f'Client: {client.ID:<10} test_acc: {client.test_Acc[-1]:<8.2f}')

    avg_test_Acc = np.mean([client.test_Acc for client in clients], axis=0)
    return avg_test_Acc




