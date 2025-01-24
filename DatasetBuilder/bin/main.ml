(*open Lexing*)
open Lexer
open Parser
open Song

let rec print_music xs = 
  match xs with
  | Song.Note n :: xs -> print_string n; print_char ' '; print_music xs
  | [] -> print_newline ()



(* print productions *)
let rec print_notes = function
  | [] -> () 
  | Note n :: rest -> print_string n; print_newline (); print_notes rest
let rec print_bars = function
  | [] ->  ()
  | Bar ns :: rest -> print_endline "Bar"; print_notes ns; print_bars rest
let print_song = function 
  | Song bs -> print_endline "Song"; print_bars bs

let song = sheet token (Lexing.from_channel stdin)
let _ = print_song (snd song)