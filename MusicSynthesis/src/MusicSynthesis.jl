module MusicSynthesis
export basic_grammar, PythonModules, sampler, uniform_prob

using Herb
using HerbGrammar, HerbSearch
using PortMidi
using PyCall
using Distributions, Random

include("./config.jl")
include("./grammar.jl")
include("./sampler.jl")
include("./play.jl")

end # module MusicSynthesis
