{
  open Lexing
  open Parser
  open Song
  type token = Parser.token
}

let digits = ['0'-'9']+
let whitespace = [' ' '\t' '\r']+
let note = ['^' '=' '_']?(['A'-'G'](','*) | (['a' - 'g'] ('\''*)))('/'?)['2' '4' '8']?
let any_letter = ['a'-'z' 'A'-'Z']
let cfg_chars = ['a'-'z' 'A'-'Z' ' ' '/' '0'-'9']+

rule token = parse
  | ['\n']
  {new_line lexbuf; token lexbuf}
  | eof
  {EOF}
  | "[M:" digits "/" digits "]" {token lexbuf}
  | "|]" {FINISH_BAR}
  | "||" {FINISH_BAR}
  | "|"  {BAR}
  | "|1"  {BAR}
  | "|:"  {BAR}
  | ":|2"  {BAR}
  | "::"  {BAR}
  | whitespace {token lexbuf}
  | any_letter as k ":" cfg_chars as v {CONFIG_LINE((k,v))}

  | ('"')? 
    (['^' '=' '_']? as accidental)
    ((['A'-'G'] (','*) | ['a'-'g'] ('\''*)) as letter_with_octave)
    (('/'?) as slash)
    (['2' '4' '8']? as len)
    ('m'?)
    ['0' - '9']*
    ('"')?
    ('>')?
    {
      let l = match slash, len with
        | "", "" -> false, 1
        | "", v -> false, int_of_string v
        | _ , "" -> true, 1
        | _, v -> true, int_of_string v
      in
      let accidental = match accidental with
        | "" -> None
        | v -> Some v.[0]
      in
      NOTE(Note(accidental, letter_with_octave.[0], (String.length letter_with_octave)-1, l))
    }

  