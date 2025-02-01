basic_grammar = HerbGrammar.@cfgrammar begin
    Start  = ("Song\n", Song)
    Song   = ("",)
    Song   = ("Bar\n", Bar, Song)
    Bar    = ("",)
    Bar    = (Note, Bar)
    Note   = ("Note ", Pitch, "\n", Length, "\n")
    Pitch  = |(-25:25)
    Length = ("Len ", LenVal)
    Length = ("Len /", LenVal)
    LenVal = |([1, 2, 4, 8])
end

const gh = Hole(BitVector())

# simple deterministic automaton that returns possible transitions
function basic_parser(cur::Vector{String})
    res = falses(61) # size of grammar - 2 productions to empty string
    if length(cur) == 0
        res[1] = true
        return res
    end
    last = cur[end]
    if startswith(last, "Song")
        res[2] = true
        return res
    end
    if startswith(last, "Bar")
        for i in 3:53
            res[i] = true
        end
        return res
    end
    if startswith(last, "Note")
        for i in 54:61
            res[i] = true
        end
        return res
    end
    if startswith(last, "Len")
        res[2] = true
        for i in 3:53
            res[i] = true
        end
        return res
    end
    return res
end
test = ["Song", "Bar", "Note 1", "Len 1", "Note 2", "Len 2", "Bar", "Note 3", "Len 3"]
test2 = ["Song", "Bar", "Note 1", "Len 1", "Bar", "Note 3", "Len 3"]