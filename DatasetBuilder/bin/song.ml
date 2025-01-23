type note = Note of char
type song = note list

type rel_note = RelNote of int
type rel_song = rel_note list

module SongConfig = Map.Make(Char)