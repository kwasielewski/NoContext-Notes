using MusicSynthesis
m = MusicSynthesis.PythonModules()
models = MusicSynthesis.setup_models(m)
pb = MusicSynthesis.playback_init()
MusicSynthesis.live_play(pb, m,  models, MusicSynthesis.model_estimator, ["Song", "Bar"], 20)
sleep(1)
MusicSynthesis.playback_stop(pb[])