import numpy as np
from . import myPlayers
from . import myModels
from . import myUtils



def run(distributed_dataset, num_classes, name_classes, public_data, args):

    FM, processor, tokenizer = myModels.load_clip_model(args)
    server_model = myModels.Image_prompting_plus_Fm(FM, processor, tokenizer, num_classes, name_classes, args).to(args.device)

    clients = [ myPlayers.Device( id, distributed_dataset[id], num_classes, name_classes, public_data, args ) for id in range(args.num_clients) ]
    clients = clients[:min(len(clients), 3)] #only three of them is enough

    server = myPlayers.Server(clients, server_model, public_data, args)


    zero_shot_logits = server.zero_shot(
        public_data["train"], 
        FM,
        processor,
        tokenizer,
        proto = True if "proto" in args.setup else False,)


    for round in range(args.rounds):
        print("=" * 20, f" Round {round + 1}/{args.rounds} ", "=" * 20)
        
        for client in clients:
            client.local_distillation(
                client.public_data,
                zero_shot_logits, 
                proto = True if "proto" in args.setup else False,
                )
            client.test_Acc.append( myUtils.Evaluate(client.model,  client.data["test"]["image"], client.data["test"]["label"], args.device)[0] )

    avg_test_Acc = np.mean([client.test_Acc for client in clients], axis=0)
    return avg_test_Acc






    