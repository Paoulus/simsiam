def log_params_to_file(model,output_file):
    for name,_ in model.named_parameters():
        print(name,file=output_file)

def log_model_structure_to_file(model,output_file):
    for idx, m in enumerate(model.modules()):
        print(idx, '->', m,file=output_file)