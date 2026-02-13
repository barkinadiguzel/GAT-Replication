class GATConfig:
    num_layers = 2
    num_heads = 8
    hidden_dim = 8
    output_dim = None  

    negative_slope = 0.2 
    dropout = 0.6

    lr = 0.005
    weight_decay = 5e-4

    add_self_loops = True
    sparse_attention = True

    seed = 42
