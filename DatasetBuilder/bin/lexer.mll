{
  open Lexing
  open Parser
  type token = Parser.token
}

let digits = ['0'-'9']+
let whitespace = [' ' '\t' '\r']+
let note = ['A'-'G' 'a' - 'g' ]
let any_letter = ['a'-'z' 'A'-'Z']
let cfg_chars = ['a'-'z' 'A'-'Z' ' ' '/' '0'-'9']+

rule token = parse
  | ['\n']
  {new_line lexbuf; token lexbuf}
  | eof
  {EOF}
  | whitespace {token lexbuf}
  | any_letter as k ":" cfg_chars as v {CONFIG_LINE((k,v))}
  | note as v {NOTE(Song.Note(v))}