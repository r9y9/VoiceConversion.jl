# Julia version of cmu_arctic_demo.sh.

JULIABIN_PATH=get(ENV, "JULIABIN_PATH", "julia")

BIN_DIR = joinpath(Pkg.dir("VoiceConversion"), "bin")
JULIABIN = [JULIABIN_PATH, "--depwarn=no", "--color=yes"]

## Experimental conditions

src_id = "clb" # Source speaker identifier
tgt_id = "slt" # Target speaker identifier

diff = true # enable direct waveform modification based on spectrum differencial
max = 100   # maximum number of training data
order = 40  # order of mel-cepstrum (expect for 0th)
mix = 8     # number of mixtures of GMMs
power_threshold = -14.0 # power threshold to select frames used in training
niter = 30  # number of iteration in EM algorithm

skip_feature_extraction = false
skip_feature_alignment = false
skip_model_training = false
skip_voice_conversion = false

## Directory settings

cmu_arctic_root = joinpath(expanduser("~"), "data", "cmu_arctic")

feature_save_dir = "./features"
model_save_dir = "./models"
vc_save_dir_top = "./converted_wav"

if diff
    vc_save_dir = joinpath(vc_save_dir_top,
                           "$(src_id)_to_$(tgt_id)_order$(order)_gmm$(mix)_diff")
else
    vc_save_dir = joinpath(vc_save_dir_top,
                           "$(src_id)_to_$(tgt_id)_order$(order)_gmm$(mix)")
end

run(`mkdir -p $(feature_save_dir)`)
run(`mkdir -p $(model_save_dir)`)
run(`mkdir -p $(vc_save_dir)`)

## Feature Extraction

if !skip_feature_extraction
    for s in [src_id, tgt_id]
        tgt_wav_path = joinpath(cmu_arctic_root, "cmu_us_$(s)_arctic/wav")
        options = ["--max", max, "--order", order]
        cmd = `$(JULIABIN) $(BIN_DIR)/mcep.jl
        $(tgt_wav_path) $(feature_save_dir)/speakers/$(s) $options`
        run(cmd)
    end
end

## Alignment

if !skip_feature_alignment
    options = ["--max", max, "--threshold", power_threshold]
    run(`$(JULIABIN) $(BIN_DIR)/align.jl
        $(feature_save_dir)/speakers/$(src_id)
        $(feature_save_dir)/speakers/$(tgt_id)
        $(feature_save_dir)/parallel/$(src_id)_and_$(tgt_id) $options`)
end

if diff
    model_path = joinpath(model_save_dir,
                          "$(src_id)_to_$(tgt_id)_gmm$(mix)_order$(order)_diff.jld")
else
    model_path = joinpath(model_save_dir,
                          "$(src_id)_to_$(tgt_id)_gmm$(mix)_order$(order).jld")
end

if !skip_model_training
    options = ["--max", max, "--n_components", mix, "--n_iter", niter,
               "--n_init", 1]
    diff && push!(options, "--diff")
    cmd = `$(JULIABIN) $(BIN_DIR)/train_gmm.jl
    $(feature_save_dir)/parallel/$(src_id)_and_$(tgt_id)
    $(model_path) $options`
    run(cmd)
end

## Voice Conversion

if !skip_voice_conversion
    synthesis_script = diff ? ["diffvc.jl"] : ["vc.jl"]
    for n=100:105
        tgt_wav_path = joinpath(cmu_arctic_root,
                                "cmu_us_$(src_id)_arctic/wav/arctic_a0$(n).wav")
        cmd = `$(JULIABIN) $(BIN_DIR)/$(synthesis_script)
        $(tgt_wav_path)
        $model_path
        $(vc_save_dir)/arctic_a0$(n)_$(src_id)_to_$(tgt_id).wav --order $order`
        run(cmd)
    end
end
