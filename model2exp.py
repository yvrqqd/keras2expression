def sigmoid(x: str) -> str:
    return f"(1/(1+exp(-({x}))))"
    
def tanh(x: str) -> str:
    return f"((exp({x})-exp(-({x})))/(exp({x})+exp(-({x}))))"

def relu(x: str) -> str:
    return f"max({x}, 0)"

def linear(x: str) -> str:
    return f"{(x)}"

def parse(model: keras.engine.functional.Functional) -> str:
    # Input vars should be declared explicitly
    ins = ["ro_f, ro_sf"]
    
    for ln, layer in enumerate(model.layers[0:]):
        try:
            weights, biases = layer.get_weights()
        except:
            # print(f"on layer {layer} mb no biases")
            continue
        if len(weights) != len(ins):
            # print(f"ins have wrong dim on {ln} layer")
            pass

        act_str = str(layer.activation)[9:-19]
        if "sigmoid" in act_str: activation = sigmoid
        elif "relu" in act_str: activation = relu
        elif "tanh" in act_str: activation = tanh
        else: activation = linear
        ins_tmp = []
        
        for wi in range(len(weights[0])):
            tmp = "("
            for n, i in enumerate(ins):
                try:
                    tmp = tmp+f"{weights[0][wi][n]}*{i}+"
                except:
                    tmp = tmp+f"{weights[0][wi]}*{i}+"
            tmp = tmp+f"{biases[wi]})"
            ins_tmp.append(activation(tmp))
        
        ins = ins_tmp
    return ins[0]
        
equation = parse(model)
