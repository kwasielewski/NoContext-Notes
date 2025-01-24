type note = Note of string
type bar  = Bar of note list
type song = Song of bar list


type rel_note = RelNote of int
type rel_bar  = RelBar of rel_note list
type rel_song = RelSong of rel_bar list

module SongConfig = Map.Make(Char)

let rel_song_of_song cfg s =
  ()