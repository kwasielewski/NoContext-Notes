%{
  open Song
  let empty_cfg : string SongConfig.t = SongConfig.empty
%}
%start <string SongConfig.t * song> sheet

%token <Song.note> NOTE
%token <char * string> CONFIG_LINE
%token BAR
%token FINISH_BAR
%token EOF
%%

sheet:
  | c = cfg_lines bs = bars EOF {(c, Song bs)}

cfg_lines : 
  | {empty_cfg}
  | cl = CONFIG_LINE rest = cfg_lines {SongConfig.add (fst cl) (snd cl) rest}

bars:
  | n = notes FINISH_BAR
    {[Bar n]}
  | n = notes BAR rest = bars
    { (Bar n) :: rest}
notes:
  | {[]}
  | n = NOTE rest = notes {n :: rest}
