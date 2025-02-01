Note_ON = 0x90
Note_OFF = 0x80
C4 = 60
base_duration = 0.3

function playback_init()
  ENV["ALSA_CONFIG_PATH"] = "/usr/share/alsa/alsa.conf"
  Pm_Initialize()
  stream = Ref{Ptr{PortMidi.PortMidiStream}}(C_NULL)
  id = 0
  Pm_OpenOutput(stream, id, C_NULL, 0, C_NULL, C_NULL, 0)
  return stream
end

function play_note(stream, note, velocity, duration)
  Pm_WriteShort(stream[], 0, Pm_Message(Note_ON, note, velocity))
  sleep(duration)
  Pm_WriteShort(stream[], 0, Pm_Message(Note_OFF, note, velocity))
end

function playback_stop(stream)
  Pm_Close(stream)
end

function truncate(cur::Vector{String}, l)
  if length(cur) > l
    return cur[end-l:end]
  end
  return cur
end

function live_play(stream, modules, models, prob_estimator, cur, notes_count)
  notes_played = 0
  block_size = 25
  for i in 1:notes_count
    cur = truncate(cur, block_size)
    sampler!(modules, models, prob_estimator, cur)
    println(cur[end])
    while !startswith(cur[end], "Len")
      cur = truncate(cur, block_size)
      sampler!(modules, models, prob_estimator, cur)
      println(cur[end])
    end
    pitch = split(cur[end-1])[2] |> x-> parse(Int, x)
    len_mult = split(cur[end])[2]
    len_mult_val = 1.0
    if len_mult[1] == '/'
      len_mult_val = base_duration / (parse(Int, len_mult[2:end]))
    else
      len_mult_val = base_duration * (parse(Int, len_mult))
    end
    @async play_note(stream, C4+pitch, 80, len_mult_val)
    sleep(len_mult_val)
  end 
end