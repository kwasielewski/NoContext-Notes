basic_grammar = HerbGrammar.@cfgrammar begin
    Start  = ("Song\n", Song)
    Song   = ("",)
    Song   = ("Bar\n", Bar, Song)
    Bar    = ("",)
    Bar    = (Note, Bar)
    Note   = (Pitch, "\n", Length, "\n")
    Pitch  = |(-12:12)
    Length = ("Len ", LenVal)
    Length = ("Len /", LenVal)
    LenVal = |([1, 2, 4, 8])
end