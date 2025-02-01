struct PythonModules
    torch :: PyCall.PyObject 
    transformers :: PyCall.PyObject
    #cuda
    trainer :: PyCall.PyObject
    tokenizer :: PyCall.PyObject
end

function PythonModules()
    torch = pyimport("torch")
    #cuda = pyimport("cuda")
    transformers = pyimport("transformers") 
    pushfirst!(PyVector(pyimport("sys")."path"), "")
    trainer = pyimport("training")
    tokenizer = trainer.AbcTokenizer()

    return PythonModules(torch, transformers, trainer, tokenizer)
end



struct Models
    dataset
    model
end

function setup_models(m::PythonModules)
    train_dataset, test_dataset = m.trainer.init_dataset("../good_run/")
    model, _ = m.trainer.init_model(train_dataset)
    model.load_state_dict(m.torch.load("./model/final", map_location=m.torch.device("cpu")))
    model.eval()
    return Models(train_dataset, model)
end
