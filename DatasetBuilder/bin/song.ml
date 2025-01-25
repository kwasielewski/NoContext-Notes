(* absolute notes *)
type accidental = char option
type octave_shift = int
type length = bool * int
type note = Note of accidental * char * octave_shift * length
type bar = Bar of note list
type song = Song of bar list

(* notes relative to key *)
type rel_note = RelNote of int * length
type rel_bar = RelBar of rel_note list
type rel_song = RelSong of rel_bar list

module SongConfig = Map.Make (Char)

let note_to_semitone = function
  | 'C' -> 0
  | 'D' -> 2
  | 'E' -> 4
  | 'F' -> 5
  | 'G' -> 7
  | 'A' -> 9
  | 'B' -> 11
  | 'c' -> 12
  | 'd' -> 14
  | 'e' -> 16
  | 'f' -> 17
  | 'g' -> 19
  | 'a' -> 21
  | 'b' -> 23
  | c -> failwith @@ "Invalid note" ^ (String.make 1 c)

let is_uppercase c = Char.uppercase_ascii c = c

let rel_song_of_song (cfg : string SongConfig.t) (s : song) : rel_song =
  let key = SongConfig.find 'K' cfg in
  let offset = note_to_semitone key.[2] (*to do sharps and flats*) in
  let offset =
    if String.length key > 3 then
      if key.[4] == '#' then offset + 1 else offset - 1
    else offset
  in
  let map_note = function
    | Note (a, c, octave_shift, l) ->
        let modifier = octave_shift * if is_uppercase c then -12 else 12 in
        let modifier =
          modifier
          +
          match a with
          | None -> 0
          | Some '^' -> 1
          | Some '_' -> -1
          | Some '=' -> failwith "naturals uninmplemented"
          | Some c -> failwith @@ "incorrect accidental" ^ String.make 1 c
        in
        RelNote
          (modifier - offset + note_to_semitone (Char.uppercase_ascii c), l)
  in
  let map_bar = function Bar x -> RelBar (List.map map_note x) in
  match s with Song s -> RelSong (List.map map_bar s)
