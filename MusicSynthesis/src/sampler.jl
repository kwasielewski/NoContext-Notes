function temp_softmax(logits; temperature=1.2)
    return softmax(logits ./ temperature)
end

function top_k_sample(probs; k=10)
    sorted = sort(probs, rev = true)
    indexes = partialsortperm(probs, 1:k, rev=true)
    index = sample(indexes, ProbabilityWeights(sorted[1:k]), 1)
    return index
end

function replace_hole(r::RuleNode)
end



function sampler(modules::PythonModules, models::Models, prob_estimator, cur::Vector{String})
    # feed the current series of tokens to network
    # get the probability
    # normalize by legal moves
    #get random move
    #later optimize building of the tree
    text_input = join(cur,"\n")
    encoded = modules.tokenizer.encode(text_input)
    #current grammar accepts almost all tokens 
    
    probs = prob_estimator(cur, modules, models)
    legal = basic_parser(cur)
    
    probs = [if legal[i] probs[i] else 0 end for i in eachindex(probs)]
    sumprob = sum(probs)
    probs ./= sumprob

    dist = Categorical(probs)
    idx = rand(dist)
    return idx
    #=
    ids = encoded.onehots
    for i in 1:max_length
        input = (; token = encoded) |> todevice
        outputs = model(input)
        logits = @view outputs.logit[:, end, 1]
        probs = temp_softmax(logits; temperature)
        new_id = top_k_sample(collect(probs); k)[1]
        push!(ids, new_id)
        new_id == ends_id && break
    end
    return decode_text(textenc, encoded)
    =#

end

function sampler!(modules::PythonModules, models::Models, prob_estimator, cur::Vector{String})
    idx = sampler(modules, models, prob_estimator, cur)
    decoded = modules.tokenizer.token_to_abc[idx-1] 
    push!(cur, decoded)
    return cur
end

function model_estimator(prods::Vector{String}, modules, models)
    input_seq = map(x->modules.tokenizer.abc_to_token[x], prods)
    output = models.model(modules.torch.tensor([input_seq]))
    probs = temp_softmax(output[1][1][1].detach().numpy())
end

function uniform_prob(prods::Vector{String}, l)
    return [1/l for _ in 1:l]
end
function uniform_estimator(p, _, _)
    return uniform_prob(p, 61)
end