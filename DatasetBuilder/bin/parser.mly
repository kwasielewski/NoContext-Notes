%{
  open Song
  let empty_cfg : string SongConfig.t = SongConfig.empty
%}
%start <string SongConfig.t * song> sheet

%token <Song.note> NOTE
%token <char * string> CONFIG_LINE
%token EOF
%%

sheet:
  | c = cfg_lines ns = notes EOF {(c, ns)}

cfg_lines : 
  | {empty_cfg}
  | cl = CONFIG_LINE rest = cfg_lines {SongConfig.add (fst cl) (snd cl) rest}

notes:
  | {[]}
  | n = NOTE rest = notes {n :: rest}
