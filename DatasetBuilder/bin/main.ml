(*open Lexing*)
open Lexer
open Parser
open Song

(* print productions *)
let rec print_notes = function
  | [] -> ()
  | Note (a, n, i, l) :: rest ->
      print_string "Note ";
      (match a with Some v -> print_char v | _ -> ());
      print_char n;
      print_newline ();
      if i != 0 then (
        print_string "Shift ";
        print_int i;
        print_newline ());
      print_string "Len ";
      if fst l then print_char '/';
      print_int (snd l);
      print_newline ();
      print_notes rest

let rec print_bars = function
  | [] -> ()
  | Bar ns :: rest ->
      print_endline "Bar";
      print_notes ns;
      print_bars rest

let print_song = function
  | Song bs ->
      print_endline "Song";
      print_bars bs


let rec print_rel_notes = function
| [] -> ()
| RelNote (i, l) :: rest ->
    print_string "Note ";
    print_int i;
    print_newline ();
    print_string "Len ";
    if fst l then print_char '/';
    print_int (snd l);
    print_newline ();
    print_rel_notes rest

let rec print_rel_bars = function
| [] -> ()
| RelBar ns :: rest ->
    print_endline "Bar";
    print_rel_notes ns;
    print_rel_bars rest

let print_rel_song = function
| RelSong bs ->
    print_endline "Song";
    print_rel_bars bs
let song = sheet token (Lexing.from_channel stdin)
(*let _ = print_song (snd song)*)

let _ = print_rel_song (rel_song_of_song (fst song) (snd song))
