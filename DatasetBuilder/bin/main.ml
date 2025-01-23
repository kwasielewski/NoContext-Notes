(*open Lexing*)
open Lexer
open Parser

let rec print_music xs = 
  match xs with
  | Song.Note c :: xs -> print_char c; print_char ' '; print_music xs
  | [] -> print_newline ()


(* print productions *)
let  _print_structure _t = 
  ()  

let song = sheet token (Lexing.from_channel stdin)
let _ = print_music (snd song)