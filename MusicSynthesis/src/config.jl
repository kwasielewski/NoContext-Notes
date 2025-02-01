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
    trainer = pyimport("demo_mul")
    tokenizer = trainer.AbcTokenizer()
    return PythonModules(torch, transformers, trainer, tokenizer)
end



struct Models
    model
end

function setup_models()
    
end
